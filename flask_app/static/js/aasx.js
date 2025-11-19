/**
 * JavaScript for AASX Generation tab with evidence-aware editors.
 */

document.addEventListener('DOMContentLoaded', () => {
    const extractSubmodelsBtn = document.getElementById('extractSubmodelsBtn');
    const generateAasxBtn = document.getElementById('generateAasxBtn');
    const submodelCheckboxes = document.querySelectorAll('.submodel-checkbox');
    const aasxStatus = document.getElementById('aasxStatus');
    const aasxProgressContainer = document.getElementById('aasxProgressContainer');
    const aasxProgressBar = document.getElementById('aasxProgressBar');
    const aasxProgressText = document.getElementById('aasxProgressText');
    const downloadContainer = document.getElementById('downloadContainer');
    const downloadLink = document.getElementById('downloadLink');
    const xmlPreview = document.getElementById('xmlPreview');
    const xmlPreviewCard = document.getElementById('xmlPreviewCard');
    const submodelEditors = document.getElementById('submodelEditors');

    const MAX_FIELDS_PER_TAB = 30;
    const getDomSafeId = (key = '') => key.replace(/[^A-Za-z0-9_-]/g, '_');
    const getEditorElement = (key) => document.getElementById(`editor_${getDomSafeId(key)}`);

    let extractedData = {};

    // Enable/disable generate button based on checkbox selection
    submodelCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const anyChecked = Array.from(submodelCheckboxes).some(cb => cb.checked);
            generateAasxBtn.disabled = !anyChecked || Object.keys(extractedData).length === 0;
            downloadContainer.style.display = 'none';
            if (xmlPreviewCard) xmlPreviewCard.style.display = 'none';
        });
    });

    // Extract submodels
    if (extractSubmodelsBtn) {
        extractSubmodelsBtn.addEventListener('click', async () => {
            const selectedSubmodels = Array.from(submodelCheckboxes)
                .filter(cb => cb.checked)
                .map(cb => cb.value);

            if (selectedSubmodels.length === 0) {
                window.pdfkg.showNotification('Please select at least one submodel', 'warning');
                return;
            }

            const llmProvider = document.querySelector('input[name="aasxLlmProvider"]:checked').value;

            extractSubmodelsBtn.disabled = true;
            extractSubmodelsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Extracting...';

            aasxProgressContainer.style.display = 'block';
            updateAasxProgress(10, 'Extracting submodels...');

            try {
                const data = await window.pdfkg.apiRequest('/extract-submodels', {
                    method: 'POST',
                    body: JSON.stringify({
                        submodels: selectedSubmodels,
                        llm_provider: llmProvider
                    })
                });

                updateAasxProgress(100, 'Extraction complete!');

                if (data.success) {
                    // Normalize metadata defaults and store originals for revert/format helpers
                    extractedData = normalizeExtraction(data.extracted || {});
                    Object.keys(extractedData).forEach(key => {
                        extractedData[key].originalData = JSON.parse(JSON.stringify(extractedData[key].data));
                    });

                    renderSubmodelEditors(extractedData);
                    generateAasxBtn.disabled = false;

                    let statusHtml = '<div class="alert alert-success mb-2">✅ Extraction complete!</div>';
                    if (data.stats && data.stats.length > 0) {
                        statusHtml += '<div class="card"><div class="card-body"><h6>Statistics:</h6><ul class="mb-0">';
                        data.stats.forEach(stat => statusHtml += `<li>${stat}</li>`);
                        statusHtml += '</ul></div></div>';
                    }
                    aasxStatus.innerHTML = statusHtml;
                    window.pdfkg.showNotification('Submodels extracted successfully!', 'success');
                } else {
                    throw new Error(data.error || 'Extraction failed');
                }
            } catch (error) {
                console.error('Extraction error:', error);
                aasxStatus.innerHTML = `<div class="alert alert-danger">❌ Error: ${error.message}</div>`;
                window.pdfkg.showNotification(`Extraction error: ${error.message}`, 'danger');
            } finally {
                extractSubmodelsBtn.disabled = false;
                extractSubmodelsBtn.innerHTML = '<i class="fas fa-brain"></i> Extract Selected Submodels';

                setTimeout(() => {
                    aasxProgressContainer.style.display = 'none';
                    updateAasxProgress(0, '');
                }, 1500);
            }
        });
    }

    // Generate AASX
    if (generateAasxBtn) {
        generateAasxBtn.addEventListener('click', async () => {
            const llmProvider = document.querySelector('input[name="aasxLlmProvider"]:checked').value;
            const selectedSubmodels = Array.from(submodelCheckboxes)
                .filter(cb => cb.checked)
                .map(cb => cb.value);

            const submodelData = {};
            for (const key of selectedSubmodels) {
                if (!extractedData[key]) continue;
                const editorTextarea = getEditorElement(key);
                if (editorTextarea) {
                    try {
                        submodelData[key] = JSON.parse(editorTextarea.value);
                    } catch (error) {
                        console.error(`Invalid JSON for ${key}:`, error);
                        window.pdfkg.showNotification(`Invalid JSON for ${key}`, 'danger');
                        return;
                    }
                }
            }

            if (Object.keys(submodelData).length === 0) {
                window.pdfkg.showNotification('No submodel data available', 'warning');
                return;
            }

            generateAasxBtn.disabled = true;
            generateAasxBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

            aasxProgressContainer.style.display = 'block';
            updateAasxProgress(50, 'Generating AAS XML...');

            try {
                const data = await window.pdfkg.apiRequest('/generate-aasx', {
                    method: 'POST',
                    body: JSON.stringify({
                        submodels: submodelData,
                        llm_provider: llmProvider
                    })
                });

                updateAasxProgress(100, 'Generation complete!');

                if (data.success) {
                    downloadLink.href = `/api/download/${data.filename}`;
                    downloadLink.textContent = `Download ${data.filename}`;
                    downloadContainer.style.display = 'block';
                    aasxStatus.innerHTML = `
                        <div class="alert alert-success mb-2">
                            <h5 class="mb-1">✅ AAS File Generated!</h5>
                            <div>File: ${data.filename}</div>
                            <div>Size: ${(data.size / 1024).toFixed(2)} KB</div>
                        </div>
                    `;

                    // Best-effort preview
                    loadXmlPreview(data.filename);
                    window.pdfkg.showNotification('AASX generated successfully!', 'success');
                } else {
                    throw new Error(data.error || 'Generation failed');
                }
            } catch (error) {
                console.error('Generation error:', error);
                aasxStatus.innerHTML = `<div class="alert alert-danger">❌ Error: ${error.message}</div>`;
                window.pdfkg.showNotification(`Generation error: ${error.message}`, 'danger');
            } finally {
                generateAasxBtn.disabled = false;
                generateAasxBtn.innerHTML = '<i class="fas fa-file-export"></i> Generate AASX';

                setTimeout(() => {
                    aasxProgressContainer.style.display = 'none';
                    updateAasxProgress(0, '');
                }, 1500);
            }
        });
    }

    function updateAasxProgress(percent, text) {
        if (aasxProgressBar) {
            aasxProgressBar.style.width = `${percent}%`;
            aasxProgressBar.textContent = `${percent}%`;
        }
        if (aasxProgressText) {
            aasxProgressText.textContent = text;
        }
    }

    function normalizeExtraction(extracted) {
        const result = {};
        Object.entries(extracted || {}).forEach(([key, payload]) => {
            const data = payload.data || {};
            const metadata = payload.metadata || {};
            Object.keys(metadata).forEach(path => {
                const entry = metadata[path] || {};
                if (entry.original_value === undefined) {
                    entry.original_value = entry.value ?? null;
                }
                if (entry.is_edited === undefined) {
                    entry.is_edited = false;
                }
                if (!entry.target_key) {
                    entry.target_key = 'value';
                }
                metadata[path] = entry;
            });
            result[key] = { data, metadata };
        });
        return result;
    }

    function categorizeMetadata(meta) {
        const needsReview = [];
        const userDefined = [];
        const completed = [];

        Object.entries(meta || {}).forEach(([path, info]) => {
            if (info.is_edited) {
                userDefined.push([path, info]);
                return;
            }
            const confidence = info.confidence || 0.0;
            const value = info.value;
            const isMissing = (value === null || value === undefined) && confidence === 0.0;
            if (isMissing || confidence < 0.8) {
                needsReview.push([path, info]);
            } else {
                completed.push([path, info]);
            }
        });

        return {
            review: needsReview.slice(0, MAX_FIELDS_PER_TAB),
            user: userDefined.slice(0, MAX_FIELDS_PER_TAB),
            completed: completed.slice(0, MAX_FIELDS_PER_TAB)
        };
    }

    function renderSubmodelEditors(extracted) {
        if (!submodelEditors) return;
        submodelEditors.innerHTML = '';

        Object.entries(extracted).forEach(([key, payload]) => {
            const card = document.createElement('div');
            card.className = 'card mb-3 submodel-editor';

            const safeKey = getDomSafeId(key);
            const navId = `nav-${safeKey}`;
            const jsonTabId = `json-${safeKey}`;
            const evidenceTabId = `evidence-${safeKey}`;

            card.innerHTML = `
                <div class="card-header d-flex justify-content-between align-items-center">
                    <div><i class="fas fa-file-alt"></i> ${key}</div>
                    <button type="button" class="btn btn-outline-secondary btn-sm" data-submodel="${key}" data-action="download-json">
                        <i class="fas fa-download"></i> Download JSON
                    </button>
                </div>
                <div class="card-body">
                    <ul class="nav nav-tabs" id="${navId}" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#${jsonTabId}" type="button" role="tab">JSON</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#${evidenceTabId}" type="button" role="tab">Confidence & Evidence</button>
                        </li>
                    </ul>
                    <div class="tab-content pt-3">
                <div class="tab-pane fade show active" id="${jsonTabId}" role="tabpanel">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <label class="form-label mb-0" for="editor_${safeKey}">JSON Data</label>
                                <div class="btn-group btn-group-sm" role="group">
                                    <button type="button" class="btn btn-outline-secondary" data-action="format-json" data-submodel="${key}"><i class="fas fa-wand-magic-sparkles"></i> Format</button>
                                    <button type="button" class="btn btn-outline-secondary" data-action="validate-json" data-submodel="${key}"><i class="fas fa-check"></i> Validate</button>
                                    <button type="button" class="btn btn-outline-secondary" data-action="revert-json" data-submodel="${key}"><i class="fas fa-rotate-left"></i> Revert</button>
                                </div>
                            </div>
                    <textarea class="form-control font-monospace submodel-json" id="editor_${safeKey}" rows="18" data-submodel="${key}">${JSON.stringify(payload.data, null, 2)}</textarea>
                            <div class="form-text">Edit and format JSON; invalid JSON will be highlighted on validation.</div>
                        </div>
                    <div class="tab-pane fade" id="${evidenceTabId}" role="tabpanel">
                        <div class="row" id="evidence_${safeKey}"></div>
                        </div>
                    </div>
                </div>
            `;

            submodelEditors.appendChild(card);
            renderEvidenceTabs(key, payload.metadata);
        });

        // Wire download buttons
        submodelEditors.querySelectorAll('[data-action="download-json"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const submodel = btn.dataset.submodel;
                const editor = getEditorElement(submodel);
                if (!editor) return;
                const blob = new Blob([editor.value], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${submodel}.json`;
                a.click();
                URL.revokeObjectURL(url);
            });
        });

        // Format/Validate/Revert handlers
        submodelEditors.querySelectorAll('[data-action="format-json"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.submodel;
                formatJsonForKey(key);
            });
        });
        submodelEditors.querySelectorAll('[data-action="validate-json"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.submodel;
                validateJsonForKey(key, { showToast: true });
            });
        });
        submodelEditors.querySelectorAll('[data-action="revert-json"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.submodel;
                revertJsonForKey(key);
            });
        });

        // Keep extractedData in sync when users edit JSON directly
        submodelEditors.querySelectorAll('.submodel-json').forEach(textarea => {
            textarea.addEventListener('change', () => {
                const key = textarea.dataset.submodel;
                if (!key || !extractedData[key]) return;
                try {
                    const parsed = JSON.parse(textarea.value);
                    extractedData[key].data = parsed;
                } catch (err) {
                    window.pdfkg.showNotification(`Invalid JSON in ${key}`, 'danger');
                }
            });
        });
    }

    function formatJsonForKey(key) {
        const editor = getEditorElement(key);
        if (!editor) return;
        try {
            const parsed = JSON.parse(editor.value);
            editor.value = JSON.stringify(parsed, null, 2);
            window.pdfkg.showNotification(`Formatted JSON for ${key}`, 'success', 1500);
        } catch (err) {
            window.pdfkg.showNotification(`Invalid JSON in ${key}`, 'danger');
        }
    }

    function validateJsonForKey(key, opts = {}) {
        const editor = getEditorElement(key);
        if (!editor) return false;
        try {
            JSON.parse(editor.value);
            editor.classList.remove('is-invalid');
            editor.classList.add('is-valid');
            if (opts.showToast) window.pdfkg.showNotification(`JSON valid for ${key}`, 'success', 1500);
            return true;
        } catch (err) {
            editor.classList.remove('is-valid');
            editor.classList.add('is-invalid');
            if (opts.showToast) window.pdfkg.showNotification(`Invalid JSON in ${key}: ${err.message}`, 'danger');
            return false;
        }
    }

    function revertJsonForKey(key) {
        const editor = getEditorElement(key);
        if (!editor || !extractedData[key] || !extractedData[key].originalData) return;
        editor.value = JSON.stringify(extractedData[key].originalData, null, 2);
        extractedData[key].data = JSON.parse(JSON.stringify(extractedData[key].originalData));
        window.pdfkg.showNotification(`Reverted JSON for ${key}`, 'info', 1500);
        renderEvidenceTabs(key, extractedData[key].metadata);
    }

    function renderEvidenceTabs(submodelKey, metadata) {
        const container = document.getElementById(`evidence_${getDomSafeId(submodelKey)}`);
        if (!container) return;

        // Some backends return metadata as dict keyed by field; convert to arrays and cap size.
        const categories = categorizeMetadata(metadata || {});
        const totalEntries = categories.review.length + categories.user.length + categories.completed.length;
        if (totalEntries === 0) {
            container.innerHTML = `<div class="alert alert-info mb-0">No confidence/evidence metadata returned for this submodel.</div>`;
            return;
        }
        container.innerHTML = renderEvidenceTabsHtml(submodelKey, categories);

        // Debug console traces to diagnose rendering issues
        console.debug("[AASX] Rendering Evidence tabs", {
            submodel: submodelKey,
            reviewCount: categories.review.length,
            userCount: categories.user.length,
            completedCount: categories.completed.length,
            rawMetadataKeys: Object.keys(metadata || {})
        });

        // Wire save/revert buttons inside accordion panes
        container.querySelectorAll('[data-action="save"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const { submodel, category, index } = btn.dataset;
                const textarea = document.getElementById(`edit_${getDomSafeId(submodel)}_${category}_${index}`);
                if (!textarea) return;
                handleFieldSave(submodel, category, Number(index), textarea.value);
            });
        });

        container.querySelectorAll('[data-action="revert"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const { submodel, category, index } = btn.dataset;
                handleFieldRevert(submodel, category, Number(index));
            });
        });
    }

    function renderEvidenceTabsHtml(submodelKey, categories) {
        const safeKey = getDomSafeId(submodelKey);
        const tabMap = [
            { key: 'review', title: '⚠️ Needs Review', badge: 'bg-warning text-dark' },
            { key: 'user', title: '✍️ User Defined', badge: 'bg-primary' },
            { key: 'completed', title: '✅ Completed', badge: 'bg-success' },
        ];

        const tabHeaders = tabMap.map((tab, idx) => `
            <li class="nav-item" role="presentation">
                <button class="nav-link ${idx === 0 ? 'active' : ''}" data-bs-toggle="tab" data-bs-target="#evtab_${safeKey}_${tab.key}" type="button" role="tab">
                    ${tab.title} <span class="badge ${tab.badge} ms-1">${categories[tab.key].length}</span>
                </button>
            </li>
        `).join('');

        const tabBodies = tabMap.map((tab, idx) => `
            <div class="tab-pane fade ${idx === 0 ? 'show active' : ''}" id="evtab_${safeKey}_${tab.key}" role="tabpanel">
                ${renderEvidenceAccordionList(submodelKey, tab.key, categories[tab.key])}
            </div>
        `).join('');

        return `
            <ul class="nav nav-tabs" role="tablist">
                ${tabHeaders}
            </ul>
            <div class="tab-content pt-3">
                ${tabBodies}
            </div>
        `;
    }

    function renderEvidenceAccordionList(submodelKey, categoryKey, entries) {
        const safeKey = getDomSafeId(submodelKey);
        if (!entries || entries.length === 0) {
            return `<div class="text-muted small">No fields in this category.</div>`;
        }

        return entries.map(([path, info], idx) => {
            const confidence = info.confidence !== undefined && info.confidence !== null
                ? Number(info.confidence).toFixed(2)
                : 'N/A';
            const sourcesHtml = Array.isArray(info.sources) && info.sources.length > 0
                ? `<h6 class="mt-2 mb-1">Sources</h6><ul class="small mb-0">${info.sources.map(s => `<li>${s}</li>`).join('')}</ul>`
                : '<div class="text-muted small">No sources provided.</div>';
            const valuePreview = info.value !== undefined && info.value !== null
                ? `<pre class="small bg-light p-2 rounded">${JSON.stringify(info.value, null, 2)}</pre>`
                : '<div class="text-muted small">No value extracted.</div>';
            const badge = info.is_edited ? 'badge bg-primary' : (categoryKey === 'completed' ? 'badge bg-success' : 'badge bg-warning text-dark');
            const status = info.is_edited ? 'User edited' : (categoryKey === 'completed' ? 'High confidence' : 'Needs review');

            return `
                <div class="accordion accordion-flush mb-2" id="acc_${safeKey}_${categoryKey}_${idx}">
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse_${safeKey}_${categoryKey}_${idx}">
                                <div class="d-flex flex-column w-100">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <span><code>${path}</code></span>
                                        <span class="${badge} ms-2">${status}</span>
                                    </div>
                                    <small class="text-muted">Confidence: ${confidence}</small>
                                </div>
                            </button>
                        </h2>
                        <div id="collapse_${safeKey}_${categoryKey}_${idx}" class="accordion-collapse collapse">
                            <div class="accordion-body">
                                <ul class="nav nav-tabs" role="tablist">
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#view_${safeKey}_${categoryKey}_${idx}" type="button" role="tab">🔍 View</button>
                                    </li>
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#edit_${safeKey}_${categoryKey}_${idx}_tab" type="button" role="tab">✍️ Edit</button>
                                    </li>
                                </ul>
                                <div class="tab-content pt-2">
                                    <div class="tab-pane fade show active" id="view_${safeKey}_${categoryKey}_${idx}" role="tabpanel">
                                        <div class="d-flex align-items-center mb-2">
                                            <span class="${badge} me-2">${status}</span>
                                            <span class="badge bg-secondary">conf: ${confidence}</span>
                                        </div>
                                        ${valuePreview}
                                        ${sourcesHtml}
                                    </div>
                                    <div class="tab-pane fade" id="edit_${safeKey}_${categoryKey}_${idx}_tab" role="tabpanel">
                                        <label class="form-label">Edit value (JSON)</label>
                                        <textarea class="form-control font-monospace" rows="6" id="edit_${safeKey}_${categoryKey}_${idx}">${JSON.stringify(info.value, null, 2)}</textarea>
                                        <div class="d-flex gap-2 mt-2">
                                            <button type="button" class="btn btn-primary btn-sm" data-action="save" data-submodel="${submodelKey}" data-category="${categoryKey}" data-index="${idx}">Save</button>
                                            <button type="button" class="btn btn-outline-secondary btn-sm" data-action="revert" data-submodel="${submodelKey}" data-category="${categoryKey}" data-index="${idx}">Revert</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    function handleFieldSave(submodelKey, category, index, newValueStr) {
        const submodel = extractedData[submodelKey];
        if (!submodel) return;

        let parsedValue;
        try {
            parsedValue = JSON.parse(newValueStr);
        } catch (err) {
            window.pdfkg.showNotification(`Invalid JSON for ${submodelKey}`, 'danger');
            return;
        }

        const lists = categorizeMetadata(submodel.metadata);
        const entry = (lists[category] || [])[index];
        if (!entry) return;
        const [path, meta] = entry;
        const targetKey = meta.target_key || 'value';

        // Update metadata
        meta.value = parsedValue;
        meta.is_edited = true;

        // Update JSON payload
        const updated = applyFieldUpdate(submodel.data, path, parsedValue, targetKey);
        if (updated) {
            const editor = getEditorElement(submodelKey);
            if (editor) {
                editor.value = JSON.stringify(submodel.data, null, 2);
            }
            window.pdfkg.showNotification(`Saved changes for ${path}`, 'success', 1500);
        } else {
            window.pdfkg.showNotification(`Field ${path} not found in JSON`, 'warning', 2500);
        }

        renderEvidenceTabs(submodelKey, submodel.metadata);
    }

    function handleFieldRevert(submodelKey, category, index) {
        const submodel = extractedData[submodelKey];
        if (!submodel) return;

        const lists = categorizeMetadata(submodel.metadata);
        const entry = (lists[category] || [])[index];
        if (!entry) return;
        const [path, meta] = entry;
        const targetKey = meta.target_key || 'value';

        meta.value = meta.original_value ?? null;
        meta.is_edited = false;

        applyFieldUpdate(submodel.data, path, meta.original_value, targetKey);
        const editor = getEditorElement(submodelKey);
        if (editor) {
            editor.value = JSON.stringify(submodel.data, null, 2);
        }
        renderEvidenceTabs(submodelKey, submodel.metadata);
        window.pdfkg.showNotification(`Reverted ${path}`, 'info', 1500);
    }

    function applyFieldUpdate(jsonData, idShort, newContent, targetKey = 'value') {
        if (!jsonData) return false;
        const copy = jsonData;

        const updateElementByIdShort = (elements) => {
            if (!Array.isArray(elements)) return false;
            for (const element of elements) {
                if (element && element.idShort === idShort) {
                    element[targetKey] = newContent ?? null;
                    return true;
                }
                const modelType = element && element.modelType;
                if ((modelType === 'SubmodelElementCollection' || modelType === 'SubmodelElementList') && Array.isArray(element.value)) {
                    if (updateElementByIdShort(element.value)) return true;
                }
            }
            return false;
        };

        if (Array.isArray(copy.submodels)) {
            for (const submodel of copy.submodels) {
                if (updateElementByIdShort(submodel.submodelElements)) {
                    return true;
                }
            }
        }

        if (updateElementByIdShort(copy.submodelElements)) {
            return true;
        }

        return false;
    }

    async function loadXmlPreview(filename) {
        if (!xmlPreview || !xmlPreviewCard) return;
        try {
            const response = await fetch(`/api/download/${filename}`);
            const text = await response.text();
            xmlPreview.textContent = text.slice(0, 8000); // avoid huge renders
            xmlPreviewCard.style.display = 'block';
        } catch (err) {
            console.error('Failed to load XML preview', err);
            xmlPreview.textContent = 'Preview unavailable.';
            xmlPreviewCard.style.display = 'block';
        }
    }
});
