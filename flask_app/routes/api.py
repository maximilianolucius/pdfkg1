"""REST API endpoints."""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename

from flask_app.system_logger import system_logger
from pdfkg.pdf_manager import ingest_pdf, auto_ingest_directory
from pdfkg.query import answer_question
from pdfkg.template_extractor import available_submodels, extract_submodels
from pdfkg.submodel_templates import get_template
from pdfkg.aas_classifier import classify_pdfs_to_aas
from pdfkg.aas_extractor import extract_aas_data
from pdfkg.aas_validator import validate_aas_data
from pdfkg.aas_xml_generator import generate_aas_xml
from pdfkg.llm.config import resolve_llm_provider
from pdfkg import llm_stats

api_bp = Blueprint('api', __name__)


def allowed_file(filename):
    """Check if file is allowed PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


@api_bp.route('/pdfs', methods=['GET'])
def list_pdfs():
    """Get list of all processed PDFs."""
    try:
        storage = current_app.config['storage']
        pdf_list = storage.list_pdfs()

        return jsonify({
            'success': True,
            'pdfs': pdf_list,
            'count': len(pdf_list)
        })
    except Exception as e:
        system_logger.log(f"Error listing PDFs: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF file(s)."""
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'success': False, 'error': 'No files selected'}), 400

        # Get parameters
        embed_model = request.form.get('embed_model', 'sentence-transformers/all-MiniLM-L6-v2')
        max_tokens = int(request.form.get('max_tokens', 500))
        use_gemini = request.form.get('use_gemini', 'false').lower() == 'true'
        force_reprocess = request.form.get('force_reprocess', 'false').lower() == 'true'

        storage = current_app.config['storage']
        upload_folder = Path(current_app.config['UPLOAD_FOLDER'])
        upload_folder.mkdir(parents=True, exist_ok=True)

        processed_results = []
        cached_results = []
        failed_results = []
        last_successful_slug = None

        # Process each file
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = upload_folder / filename
                file.save(str(filepath))

                system_logger.log(f"Processing uploaded file: {filename}", "INFO")

                def progress_callback(pct: float, desc: str):
                    # Log coarse-grained progress for visibility
                    system_logger.log(f"[{filename}] {desc} ({int(pct * 100)}%)", "INFO")

                try:
                    # Process PDF
                    result = ingest_pdf(
                        pdf_path=filepath,
                        storage=storage,
                        embed_model=embed_model,
                        max_tokens=max_tokens,
                        use_gemini=use_gemini,
                        gemini_pages="",
                        force_reprocess=force_reprocess,
                        save_to_db=True,
                        save_files=True,
                        output_dir=None,
                        progress_callback=progress_callback
                    )

                    last_successful_slug = result.pdf_slug
                    summary = result.summary()

                    if result.was_cached:
                        cached_results.append(summary)
                    else:
                        processed_results.append(summary)

                except Exception as e:
                    system_logger.log(f"Error processing {filename}: {e}", "ERROR")
                    failed_results.append({
                        'filename': filename,
                        'error': str(e)
                    })

        return jsonify({
            'success': True,
            'processed': processed_results,
            'cached': cached_results,
            'failed': failed_results,
            'last_slug': last_successful_slug
        })

    except Exception as e:
        system_logger.log(f"Error in upload endpoint: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/query', methods=['POST'])
def query_pdf():
    """Query a PDF with a question."""
    try:
        data = request.get_json()

        question = data.get('question')
        pdf_slug = data.get('pdf_slug')
        llm_provider = data.get('llm_provider', 'gemini')
        top_k = data.get('top_k', 5)
        embed_model = data.get('embed_model', 'sentence-transformers/all-MiniLM-L6-v2')

        if not question or not pdf_slug:
            return jsonify({'success': False, 'error': 'Missing question or pdf_slug'}), 400

        storage = current_app.config['storage']

        # Get answer
        result = answer_question(
            question=question,
            pdf_slug=pdf_slug,
            model_name=embed_model,
            top_k=top_k,
            llm_provider=llm_provider,
            storage=storage,
        )

        return jsonify({
            'success': True,
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'debug': result.get('debug', {})
        })

    except Exception as e:
        system_logger.log(f"Error in query endpoint: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/reset', methods=['POST'])
def reset_project():
    """Reset project data (dangerous operation)."""
    try:
        storage = current_app.config['storage']

        # Clear output directory
        output_dir = Path("data/output")
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            system_logger.log("Cleared data/output/", "INFO")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Reset ArangoDB
        if hasattr(storage, "db_client"):
            storage.db_client.reset_database()
            system_logger.log(f"Reset ArangoDB database {storage.db_client.db_name}", "INFO")

        # Reset Milvus
        if hasattr(storage, "milvus_client") and storage.milvus_client:
            default_dim = os.getenv("DEFAULT_EMBED_DIM")
            dimension = int(default_dim) if default_dim else None
            storage.milvus_client.reset_collection(dimension=dimension)
            system_logger.log(f"Reset Milvus collection {storage.milvus_client.collection_name}", "INFO")

        # Re-ingest PDFs from data/input
        use_gemini_default = bool(os.getenv("GEMINI_API_KEY"))

        def reset_progress(pdf_name: str, status: str):
            system_logger.log(f"[reset] {pdf_name}: {status}", "INFO")

        ingest_results = auto_ingest_directory(
            input_dir=Path("data/input"),
            storage=storage,
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=500,
            use_gemini=use_gemini_default,
            save_to_db=True,
            save_files=True,
            progress_callback=reset_progress,
            force_reprocess=True,
        )

        return jsonify({
            'success': True,
            'message': 'Project reset complete',
            'ingest_results': {
                'processed': len(ingest_results.get('processed', [])),
                'cached': len(ingest_results.get('skipped', [])),
                'failed': len(ingest_results.get('failed', []))
            }
        })

    except Exception as e:
        system_logger.log(f"Error resetting project: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/submodels', methods=['GET'])
def get_submodels():
    """Get available submodels for AASX generation."""
    try:
        submodels = available_submodels()
        submodel_info = []

        for key in submodels:
            template = get_template(key)
            submodel_info.append({
                'key': key,
                'display_name': template.display_name,
                'description': template.description
            })

        return jsonify({
            'success': True,
            'submodels': submodel_info
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/extract-submodels', methods=['POST'])
def extract_submodels_api():
    """Extract submodels from PDFs."""
    try:
        data = request.get_json()
        selected_submodels = data.get('submodels', [])
        llm_provider = data.get('llm_provider', 'gemini')
        use_batch = data.get('use_batch', True)

        if not selected_submodels:
            return jsonify({'success': False, 'error': 'No submodels selected'}), 400

        storage = current_app.config['storage']

        system_logger.log(f"Extracting submodels ({'batch' if use_batch else 'legacy'}): {', '.join(selected_submodels)}", 'INFO')

        extracted = extract_submodels(
            storage=storage,
            submodels=selected_submodels,
            llm_provider=llm_provider,
            progress_callback=None,
            use_batch=use_batch
        )

        # Format results
        results = {}
        for key in selected_submodels:
            template = get_template(key)
            result = extracted.get(key)
            if result:
                results[key] = {
                    'data': result.get('data', template.schema),
                    'metadata': result.get('metadata', {})
                }
            else:
                results[key] = {
                    'data': template.schema,
                    'metadata': {}
                }

        summary_lines = list(llm_stats.summary_lines())
        for line in summary_lines:
            system_logger.log(line, 'INFO')

        return jsonify({
            'success': True,
            'extracted': results,
            'stats': summary_lines
        })

    except Exception as e:
        system_logger.log(f"Error extracting submodels: {e}", 'ERROR')
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/generate-aasx', methods=['POST'])
def generate_aasx_api():
    """Generate AASX file from submodel data."""
    try:
        data = request.get_json()
        submodel_data = data.get('submodels', {})
        llm_provider = data.get('llm_provider', 'gemini')

        if not submodel_data:
            return jsonify({'success': False, 'error': 'No submodel data provided'}), 400

        storage = current_app.config['storage']

        # Normalize provider and generate timestamped path
        llm_provider_resolved = resolve_llm_provider(llm_provider)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/out")
        output_dir.mkdir(parents=True, exist_ok=True)
        xml_path = output_dir / f"aas_output_{timestamp}.xml"

        xml_output = generate_aas_xml(
            storage=storage,
            llm_provider=llm_provider_resolved,
            output_path=xml_path,
            data=submodel_data
        )

        if not xml_output:
            return jsonify({'success': False, 'error': 'XML generation failed'}), 500

        system_logger.log(f"Generated XML at {xml_path}", 'INFO')

        return jsonify({
            'success': True,
            'xml_path': str(xml_path),
            'filename': xml_path.name,
            'size': len(xml_output)
        })

    except Exception as e:
        system_logger.log(f"Error generating AASX: {e}", 'ERROR')
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download generated file."""
    try:
        candidates = [
            Path("data/out") / filename,
            Path("data/output") / filename,
        ]
        file_path = next((p for p in candidates if p.exists()), None)

        if not file_path:
            return jsonify({'success': False, 'error': 'File not found'}), 404

        return send_file(str(file_path), as_attachment=True)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/logs', methods=['GET'])
def get_logs():
    """Get system logs."""
    try:
        logs = system_logger.get_logs_list()
        return jsonify({
            'success': True,
            'logs': logs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/logs/clear', methods=['POST'])
def clear_logs():
    """Clear system logs."""
    try:
        system_logger.clear()
        return jsonify({
            'success': True,
            'message': 'Logs cleared'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
