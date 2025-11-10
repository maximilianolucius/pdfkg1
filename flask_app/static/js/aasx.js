/**
 * JavaScript for AASX Generation tab
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
    const submodelEditors = document.getElementById('submodelEditors');

    let extractedData = {};

    // Enable/disable generate button based on checkbox selection
    submodelCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const anyChecked = Array.from(submodelCheckboxes).some(cb => cb.checked);
            generateAasxBtn.disabled = !anyChecked || Object.keys(extractedData).length === 0;

            // Hide download container when selection changes
            downloadContainer.style.display = 'none';
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

            // Disable button
            extractSubmodelsBtn.disabled = true;
            extractSubmodelsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Extracting...';

            // Show progress
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
                    extractedData = data.extracted;

                    // Display results
                    displayExtractedSubmodels(data.extracted);

                    // Enable generate button
                    generateAasxBtn.disabled = false;

                    // Update status
                    let statusHtml = '<div class="alert alert-success">✅ Extraction complete!</div>';
                    if (data.stats && data.stats.length > 0) {
                        statusHtml += '<div class="card"><div class="card-body"><h6>Statistics:</h6><ul>';
                        data.stats.forEach(stat => {
                            statusHtml += `<li>${stat}</li>`;
                        });
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
                }, 2000);
            }
        });
    }

    // Generate AASX
    if (generateAasxBtn) {
        generateAasxBtn.addEventListener('click', async () => {
            const llmProvider = document.querySelector('input[name="aasxLlmProvider"]:checked').value;

            // Collect current submodel data (including any edits)
            const submodelData = {};
            Object.keys(extractedData).forEach(key => {
                const editorTextarea = document.getElementById(`editor_${key}`);
                if (editorTextarea) {
                    try {
                        submodelData[key] = JSON.parse(editorTextarea.value);
                    } catch (error) {
                        console.error(`Invalid JSON for ${key}:`, error);
                        window.pdfkg.showNotification(`Invalid JSON for ${key}`, 'danger');
                        throw error;
                    }
                }
            });

            if (Object.keys(submodelData).length === 0) {
                window.pdfkg.showNotification('No submodel data available', 'warning');
                return;
            }

            // Disable button
            generateAasxBtn.disabled = true;
            generateAasxBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

            // Show progress
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
                    // Show download link
                    downloadLink.href = `/api/download/${data.filename}`;
                    downloadLink.textContent = `Download ${data.filename}`;
                    downloadContainer.style.display = 'block';

                    // Update status
                    aasxStatus.innerHTML = `
                        <div class="alert alert-success">
                            <h5>✅ AAS File Generated!</h5>
                            <p>File: ${data.filename}</p>
                            <p>Size: ${(data.size / 1024).toFixed(2)} KB</p>
                        </div>
                    `;

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
                }, 2000);
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

    function displayExtractedSubmodels(extracted) {
        if (!submodelEditors) return;

        submodelEditors.innerHTML = '';

        Object.keys(extracted).forEach(key => {
            const data = extracted[key];
            const jsonStr = JSON.stringify(data.data, null, 2);

            const accordionItem = document.createElement('div');
            accordionItem.className = 'accordion-item mb-3';

            // Format metadata
            let metadataHtml = '<p class="small text-muted">No evidence captured yet.</p>';
            if (data.metadata && Object.keys(data.metadata).length > 0) {
                metadataHtml = '<h6>Evidence:</h6><ul class="small">';
                Object.keys(data.metadata).forEach(path => {
                    const info = data.metadata[path];
                    const confidence = info.confidence ? info.confidence.toFixed(2) : 'N/A';
                    metadataHtml += `<li><code>${path}</code> (confidence: ${confidence})`;
                    if (info.sources && info.sources.length > 0) {
                        metadataHtml += '<ul>';
                        info.sources.slice(0, 3).forEach(source => {
                            metadataHtml += `<li>${source}</li>`;
                        });
                        metadataHtml += '</ul>';
                    }
                    metadataHtml += '</li>';
                });
                metadataHtml += '</ul>';
            }

            accordionItem.innerHTML = `
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse_${key}">
                        ${key}
                    </button>
                </h2>
                <div id="collapse_${key}" class="accordion-collapse collapse">
                    <div class="accordion-body">
                        <div class="mb-3">
                            <label for="editor_${key}" class="form-label">JSON Data</label>
                            <textarea class="form-control font-monospace" id="editor_${key}" rows="15">${jsonStr}</textarea>
                            <div class="form-text">You can edit the JSON data before generating the AAS file</div>
                        </div>
                        <div class="mb-3">
                            ${metadataHtml}
                        </div>
                    </div>
                </div>
            `;

            submodelEditors.appendChild(accordionItem);
        });

        // Initialize Bootstrap accordions
        const accordions = submodelEditors.querySelectorAll('.accordion-collapse');
        accordions.forEach(accordion => {
            new bootstrap.Collapse(accordion, { toggle: false });
        });
    }
});
