"""
Template-driven AAS submodel extraction using a "Scaffold and Fill" strategy.
Updated to use BATCHED extraction to significantly reduce token usage by
sending context once for multiple fields. Includes intelligent hint passing
from 'description' fields to guide 'value' extraction.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from pdfkg import llm_stats
from pdfkg.llm.config import resolve_llm_provider
from pdfkg.llm.mistral_client import (chat as mistral_chat,
                                      get_model_name as get_mistral_model_name)
from pdfkg.submodel_templates import get_template, list_submodel_templates

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


# Configuration
BATCH_SIZE = 8  # Number of fields to extract in a single LLM call

SUBMODEL_QUERIES: Dict[str, str] = {
    "DigitalNameplate": "manufacturer name, model number, part number, product designation, serial number, year of construction, identification, address information, manufacturing location, uri of product",
    "TechnicalData": "technical specifications, data sheet, voltage, current, power, IP rating, operating temperature, pressure, frequency, material",
    "Documentation": "installation guide, safety manual, operating instructions, user manual, datasheet, document reference",
    "HandoverDocumentation": "certificate, compliance, declaration, warranty, CE, UL, ATEX, conformity",
    "MaintenanceRecord": "maintenance schedule, interval, service procedure, spare parts, troubleshooting, repair, inspection, lubrication",
    "OperationalData": "operating parameters, settings, conditions, operating mode, startup, shutdown, commissioning, control",
    "BillOfMaterials": "bill of materials, BOM, component list, parts list, accessories, spare parts, part number, quantity",
    "CarbonFootprint": "carbon footprint, CO2 emissions, energy consumption, sustainability, lifecycle, environmental impact, recycling, eco-design",
}

def _normalize_language_code(lang_input: str) -> str:
    """
    Ensures the language code is a valid ISO 639-1 string.
    Maps common full names to codes if the LLM slips up.
    """
    if not lang_input or not isinstance(lang_input, str):
        return "en" # Default fallback
    
    clean = lang_input.lower().strip()
    
    # Common mappings if LLM returns full names
    mapping = {
        "english": "en", "german": "de", "french": "fr", 
        "spanish": "es", "italian": "it", "dutch": "nl",
        "chinese": "zh", "japanese": "ja", "russian": "ru"
    }
    
    if clean in mapping:
        return mapping[clean]
    
    # If it looks like a locale (en-US), take the first part
    if '-' in clean:
        return clean.split('-')[0]
        
    return clean[:2] # Force 2 chars if it's a standard code

@dataclass
class ExtractionJob:
    """Holds all necessary information to perform one targeted extraction."""
    id_short: str
    model_type: str
    json_path: List[Any]
    target_key: str
    cardinality: str = "ZeroToOne"
    value_type: str = "xs:string"
    item_schema: Dict = field(default_factory=dict)
    existing_description: str = ""


@dataclass
class FieldExtraction:
    """Holds the result of one completed extraction job."""
    path: str
    value: Any
    confidence: float
    sources: List[str]
    notes: str = ""
    json_path: List[Any] = field(default_factory=list)


@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    metadata: Dict[str, Dict[str, Any]]

class TemplateAASExtractor:
    """Extracts submodel data by parsing templates to create targeted, parallel jobs."""

    def __init__(self, storage, llm_provider: str = "gemini") -> None:
        self.storage = storage
        self.llm_provider = resolve_llm_provider(llm_provider) if llm_provider else resolve_llm_provider(None)
        if self.llm_provider == "gemini":
            if not GEMINI_AVAILABLE: raise ImportError("google-generativeai not installed")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY not configured")
            genai.configure(api_key=api_key)
            self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            self.llm_client = genai.GenerativeModel(self.gemini_model_name)
        elif self.llm_provider == "mistral":
            if not MISTRAL_AVAILABLE: raise ImportError("mistralai not installed")
            self.mistral_model_name = get_mistral_model_name()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def extract(self, submodels: Iterable[str], progress_callback=None) -> Dict[str, ExtractionResult]:
        pdf_slugs = [pdf["slug"] for pdf in self.storage.list_pdfs()]
        if not pdf_slugs: raise RuntimeError("No PDFs available for extraction.")
        
        results: Dict[str, ExtractionResult] = {}
        
        # Ensure submodels is a list to get its length
        submodel_list = list(submodels)
        
        for i, submodel_key in enumerate(submodel_list):
            if progress_callback: progress_callback(i / len(submodel_list), f"Planning for {submodel_key}...")
            
            template = get_template(submodel_key)
            scaffold_dict = json.loads(template.json)
            
            jobs = self._create_extraction_jobs(scaffold_dict)
            
            if not jobs:
                results[submodel_key] = ExtractionResult(data=scaffold_dict, metadata={})
                continue
                
            if progress_callback: progress_callback(i / len(submodel_list) + 0.1, f"Executing {len(jobs)} jobs in batches...")
            
            extraction_results = self._execute_batched_jobs(jobs, submodel_key, pdf_slugs)
            
            final_data, metadata = self._inject_results(scaffold_dict, extraction_results)
            results[submodel_key] = ExtractionResult(data=final_data, metadata=metadata)
            
        if progress_callback: progress_callback(1.0, "Extraction complete.")
        return results

    def _create_extraction_jobs(self, schema: Dict) -> List[ExtractionJob]:
        """
        Recursively parses a template to find all fields that need data extraction.
        Prioritizes 'value' and treats 'description' as a hint.
        """
        jobs = []
        
        def find_targets(element: Any, path: List[Any]):
            if not isinstance(element, (dict, list)):
                return

            if isinstance(element, dict):
                id_short, model_type = element.get("idShort"), element.get("modelType")
                
                if id_short:
                    qualifiers = element.get("qualifiers", [])
                    cardinality = next((q.get("value", "ZeroToOne") for q in qualifiers if q.get("type") == "SMT/Cardinality"), "ZeroToOne")

                    # Capture the description text first, to be used as a potential hint.
                    existing_desc_text = ""
                    if isinstance(element.get("description"), list) and element["description"]:
                        try:
                            existing_desc_text = element["description"][0].get("text", "")
                        except (IndexError, AttributeError):
                            existing_desc_text = ""

                    # Rule: Prioritize the 'value' key for extraction.
                    if "value" in element:
                        jobs.append(ExtractionJob(
                            id_short=id_short,
                            model_type=model_type,
                            cardinality=cardinality,
                            value_type=element.get("valueType", "xs:string"), 
                            target_key="value",
                            json_path=path + ["value"],
                            existing_description=existing_desc_text
                        ))
                    
                    # Fallback Rule: Only create a job for 'description' if 'value' is absent.
                    elif "description" in element:
                         jobs.append(ExtractionJob(
                            id_short=id_short,
                            model_type="MultiLanguageProperty",
                            cardinality="ZeroToMany",
                            value_type="xs:string",
                            target_key="description",
                            json_path=path + ["description"],
                            existing_description=existing_desc_text
                        ))

                # Recurse into all children to find more targets.
                for key, value in element.items():
                    if id_short and key in ["value", "description"]:
                        continue
                    find_targets(value, path + [key])
            
            elif isinstance(element, list):
                for i, item in enumerate(element):
                    find_targets(item, path + [i])

        find_targets(schema, [])
        return jobs

    def _create_simple_schema(self, element: Dict) -> Dict:
        simple = {}
        if not isinstance(element, dict): return {}
        elements_to_process = element.get("value", [element])
        if not isinstance(elements_to_process, list): elements_to_process = [elements_to_process]
        for item in elements_to_process:
            if isinstance(item, dict) and "idShort" in item:
                id_short = item.get("idShort")
                if id_short: simple[id_short] = self._create_simple_schema(item) if "value" in item and isinstance(item["value"], (dict, list)) else None
        return simple

    def _execute_batched_jobs(self, jobs: List[ExtractionJob], submodel_key: str, pdf_slugs: List[str]) -> List[FieldExtraction]:
        results = []
        batches = [jobs[i:i + BATCH_SIZE] for i in range(0, len(jobs), BATCH_SIZE)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {
                executor.submit(self._run_batch, batch, submodel_key, pdf_slugs): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as exc:
                    batch_jobs = future_to_batch[future]
                    print(f"ERROR processing batch starting with '{batch_jobs[0].id_short}': {exc}")
                    for job in batch_jobs:
                        results.append(FieldExtraction(
                            path=job.id_short, value=None, confidence=0.0, 
                            sources=[], notes="Batch processing failed", json_path=job.json_path
                        ))
        return results

    def _run_batch(self, jobs: List[ExtractionJob], submodel_key: str, pdf_slugs: List[str]) -> List[FieldExtraction]:
        evidence = self._collect_field_evidence(f"{submodel_key} " + " ".join([j.id_short for j in jobs]), pdf_slugs, max_snippets=10)
        prompt = self._build_batch_prompt(submodel_key, jobs, evidence)

        first_id = jobs[0].id_short
        last_id = jobs[-1].id_short
        label = f"{submodel_key}:Batch[{first_id}..{last_id}]"
        
        llm_response_text = self._query_llm(prompt, phase="extraction", label=label)

        llm_stats.log_extraction_io(
            id_short=f"BATCH_{first_id}_to_{last_id}",
            prompt=prompt,
            output=llm_response_text,
            metadata={"model": self.gemini_model_name if self.llm_provider == "gemini" else "mistral"}
        )

        parsed_json = self._parse_field_response(llm_response_text)
        
        batch_results = []
        for job in jobs:
            field_data = parsed_json.get(job.id_short, {})
            if not isinstance(field_data, dict):
                field_data = {}

            value = field_data.get("value")
            
            if (job.model_type == "MultiLanguageProperty" or job.target_key == "description") and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "language" in item:
                        item["language"] = _normalize_language_code(item.get("language", ""))
            
            if job.model_type == "SubmodelElementList" and value is None:
                 value = field_data.get("items", [])

            batch_results.append(FieldExtraction(
                path=job.id_short,
                value=value,
                confidence=float(field_data.get("confidence", 0.0)) if _is_number(field_data.get("confidence")) else 0.0,
                sources=field_data.get("sources") or [],
                notes=field_data.get("notes", ""),
                json_path=job.json_path
            ))

        return batch_results

    def _build_batch_prompt(self, submodel: str, jobs: List[ExtractionJob], evidence: List[str]) -> str:
        evidence_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(evidence))
        
        schema_desc = []
        for job in jobs:
            if job.model_type == "SubmodelElementList":
                desc = f"- '{job.id_short}': A LIST of objects. Cardinality: {job.cardinality}. Item schema: {list(job.item_schema.keys())}"
            elif job.target_key == "description" or job.model_type == "MultiLanguageProperty":
                desc = f"- '{job.id_short}': Multi-language text. Return a list: [{{ \"language\": \"ISO 639-1 code (e.g. 'en', 'de')\", \"text\": \"...\" }}]. Cardinality: {job.cardinality}."
            else:
                desc = f"- '{job.id_short}': Single {job.value_type}. Cardinality: {job.cardinality}."
            
            if job.existing_description:
                desc += f" (Hint: {job.existing_description})"

            schema_desc.append(desc)
        
        schema_str = "\n".join(schema_desc)

        return (
            f"You are extracting data for the {submodel} submodel.\n"
            f"Extract the following fields based strictly on the evidence provided:\n\n"
            f"TARGET FIELDS:\n{schema_str}\n\n"
            "INSTRUCTIONS:\n"
            "1. Return a single JSON object where keys are the field names (idShorts) listed above.\n"
            "2. For each key, the value must be an object: { \"value\": ..., \"confidence\": 0.0-1.0, \"sources\": [], \"notes\": \"\" }.\n"
            "3. For Multi-language fields, use the context to determine the language code (e.g., if the text is English, use 'en').\n"
            "4. Adhere to any hints provided for a field.\n"
            "5. If specific data is not found for a field, set its \"value\" to null and \"confidence\" to 0.0.\n"
            "6. Do not invent data.\n\n"
            f"Evidence:\n{evidence_block}\n"
        )

    def _inject_results(self, scaffold: Dict, results: List[FieldExtraction]) -> Tuple[Dict, Dict]:
        metadata = {}
        for result in results:
            current_level = scaffold
            try:
                for key in result.json_path[:-1]:
                    current_level = current_level[key]
                final_key = result.json_path[-1]
                
                if result.value is not None:
                    current_level[final_key] = result.value
            except (KeyError, IndexError, TypeError) as e:
                print(f"  WARNING: Could not inject result for idShort '{result.path}'. Path: {result.json_path}. Error: {e}")

            metadata[result.path] = {
                "confidence": result.confidence,
                "sources": result.sources,
                "notes": result.notes,
                "value": result.value 
            }
            
        return scaffold, metadata

    def _collect_field_evidence(self, field_path: str, pdf_slugs: List[str], max_snippets: int = 5) -> List[str]:
        tokens = [token for token in field_path.replace('.', ' ').replace('_', ' ').lower().split() if len(token) > 3]
        
        snippets, seen = [], set()
        for slug in pdf_slugs:
            pdf_info, chunks = self.storage.get_pdf_metadata(slug) or {}, self.storage.get_chunks(slug)
            for chunk in chunks:
                text = chunk.get("text", "")
                if not text: continue
                
                if tokens and not any(token in text.lower() for token in tokens): 
                    continue
                    
                snippet = self._format_snippet(pdf_info.get("filename", slug), chunk, text)
                if snippet in seen: continue
                snippets.append(snippet)
                seen.add(snippet)
                
                if len(snippets) >= max_snippets: return snippets
        
        if not snippets:
            context = self._build_context(pdf_slugs, field_path.split(' ')[0] if field_path else '')
            snippets.extend(b.strip() for b in context.split("\n\n") if b.strip())
            
        return snippets[:max_snippets] if snippets else ["No direct evidence found in documents."]

    def _format_snippet(self, filename: str, chunk: Dict[str, Any], text: str) -> str:
        page = chunk.get("page")
        return f"[{filename}{f' | page {page}' if page is not None else ''}] {' '.join(text.strip().split())[:800]}"

    def _build_context(self, pdf_slugs: List[str], submodel: str) -> str:
        query_hint = SUBMODEL_QUERIES.get(submodel, submodel)
        context_blocks = []
        for slug in pdf_slugs:
            pdf_info = self.storage.get_pdf_metadata(slug) or {}
            chunks = self.storage.get_chunks(slug)[:10]
            text = "\n".join(c.get("text", "") for c in chunks if c.get("text"))
            if text: context_blocks.append(f"[PDF: {pdf_info.get('filename', slug)}]\n{text}")
        return "\n\n".join(context_blocks[:10]) if context_blocks else f"No textual context found. Focus on template: {query_hint}."

    def _query_llm(self, prompt: str, phase: str, label: str) -> str:
        start_time = time.time()
        try:
            if self.llm_provider == "gemini":
                start = time.time()
                response = self.llm_client.generate_content(prompt)
                usage = getattr(response, "usage_metadata", None)
                tokens_in, tokens_out, total_tokens = llm_stats.extract_token_usage(usage)
                llm_stats.record_call(
                    self.llm_provider, phase=phase, label=label,
                    tokens_in=tokens_in or 0, tokens_out=tokens_out or 0, total_tokens=total_tokens or 0,
                    metadata={
                    "model": self.gemini_model_name,
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "prompt_chars": len(prompt),
                })
                return response.text
            elif self.llm_provider == "mistral":
                response = mistral_chat(messages=[{"role": "user", "content": prompt}], model=self.mistral_model_name)
                usage = getattr(response, "usage", None)
                tokens_in, tokens_out, total_tokens = llm_stats.extract_token_usage(usage)
                content = response.choices[0].message.content
                output_text = str(content) if isinstance(content, str) else json.dumps(content)
                
                llm_stats.record_call(
                    self.llm_provider, phase=phase, label=label,
                    tokens_in=tokens_in, tokens_out=tokens_out, total_tokens=total_tokens,
                    elapsed_ms=int((time.time() - start_time) * 1000),
                    metadata={'model': self.mistral_model_name}
                )
                return output_text
        except Exception as e:
            print(f"  ❌ LLM query failed for {label}: {e}")
            llm_stats.record_call(
                self.llm_provider, phase=phase, label=f"FAILED: {label}",
                elapsed_ms=int((time.time() - start_time) * 1000)
            )
            return ""
        return ""

    def _parse_field_response(self, response_text: str) -> Dict[str, Any]:
        if not isinstance(response_text, str): response_text = str(response_text)
        json_start, json_end = response_text.find('{'), response_text.rfind('}')
        if json_start == -1 or json_end < json_start: return {}
        json_string = response_text[json_start : json_end + 1]
        try:
            payload = json.loads(json_string)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

def extract_submodels(storage, submodels: Iterable[str], llm_provider: str = "gemini", progress_callback=None) -> Dict[str, Dict[str, Any]]:
    extractor = TemplateAASExtractor(storage=storage, llm_provider=llm_provider)
    raw_results = extractor.extract(submodels, progress_callback=progress_callback)
    return {key: {"data": result.data, "metadata": result.metadata} for key, result in raw_results.items()}

def available_submodels() -> List[str]:
    return list_submodel_templates()

def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
