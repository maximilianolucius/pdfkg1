/**
 * JavaScript for PDF Ingestion tab
 */

document.addEventListener('DOMContentLoaded', () => {
    const ingestForm = document.getElementById('ingestForm');
    const processBtn = document.getElementById('processBtn');
    const resetBtn = document.getElementById('resetBtn');
    const maxTokensSlider = document.getElementById('maxTokens');
    const maxTokensValue = document.getElementById('maxTokensValue');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const statusOutput = document.getElementById('statusOutput');

    // Update max tokens display
    if (maxTokensSlider && maxTokensValue) {
        maxTokensSlider.addEventListener('input', () => {
            maxTokensValue.textContent = maxTokensSlider.value;
        });
    }

    // Handle PDF processing
    if (ingestForm) {
        ingestForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(ingestForm);
            const files = formData.getAll('files');

            if (files.length === 0 || files[0].name === '') {
                window.pdfkg.showNotification('Please select at least one PDF file', 'warning');
                return;
            }

            // Disable form
            processBtn.disabled = true;
            processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

            // Show progress
            progressContainer.style.display = 'block';
            updateProgress(10, 'Uploading files...');

            try {
                // Upload and process PDFs
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                updateProgress(50, 'Processing PDFs...');

                const data = await response.json();

                if (data.success) {
                    updateProgress(100, 'Complete!');

                    // Format status message
                    const statusHtml = formatProcessingStatus(data);
                    statusOutput.innerHTML = statusHtml;

                    // Refresh PDF list
                    await window.pdfkg.refreshPdfList();

                    // Enable chat if PDFs were processed
                    if (data.processed.length > 0 || data.cached.length > 0) {
                        enableChat();
                    }

                    window.pdfkg.showNotification('PDF processing complete!', 'success');
                } else {
                    throw new Error(data.error || 'Processing failed');
                }
            } catch (error) {
                console.error('Processing error:', error);
                statusOutput.innerHTML = `<div class="alert alert-danger">❌ Error: ${error.message}</div>`;
                window.pdfkg.showNotification(`Error: ${error.message}`, 'danger');
            } finally {
                // Re-enable form
                processBtn.disabled = false;
                processBtn.innerHTML = '<i class="fas fa-rocket"></i> Process PDF';

                // Hide progress after delay
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    updateProgress(0, '');
                }, 2000);
            }
        });
    }

    // Handle project reset
    if (resetBtn) {
        resetBtn.addEventListener('click', async () => {
            if (!confirm('⚠️ WARNING: This will delete all processed data and reset the database. Continue?')) {
                return;
            }

            resetBtn.disabled = true;
            resetBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Resetting...';

            window.pdfkg.showSpinner('Resetting project data...');

            try {
                const data = await window.pdfkg.apiRequest('/reset', {
                    method: 'POST'
                });

                if (data.success) {
                    statusOutput.innerHTML = `
                        <div class="alert alert-success">
                            <h5>✅ Project Reset Complete!</h5>
                            <p>${data.message}</p>
                            <ul>
                                <li>Newly processed: ${data.ingest_results.processed}</li>
                                <li>Cached: ${data.ingest_results.cached}</li>
                                <li>Failed: ${data.ingest_results.failed}</li>
                            </ul>
                        </div>
                    `;

                    // Refresh PDF list
                    await window.pdfkg.refreshPdfList();

                    window.pdfkg.showNotification('Project reset complete!', 'success');
                } else {
                    throw new Error(data.error || 'Reset failed');
                }
            } catch (error) {
                console.error('Reset error:', error);
                window.pdfkg.showNotification(`Reset failed: ${error.message}`, 'danger');
            } finally {
                window.pdfkg.hideSpinner();
                resetBtn.disabled = false;
                resetBtn.innerHTML = '<i class="fas fa-recycle"></i> Reset Project Data';
            }
        });
    }

    function updateProgress(percent, text) {
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
            progressBar.textContent = `${percent}%`;
        }
        if (progressText) {
            progressText.textContent = text;
        }
    }

    function formatProcessingStatus(data) {
        const total = data.processed.length + data.cached.length + data.failed.length;

        if (total === 0) {
            return '<div class="alert alert-warning">❌ No PDFs were processed.</div>';
        }

        let html = '';

        // Header
        if (total === 1) {
            if (data.processed.length > 0) {
                html += '<div class="alert alert-success"><h5>✅ PDF processed successfully!</h5></div>';
            } else if (data.cached.length > 0) {
                html += '<div class="alert alert-info"><h5>ℹ️ PDF already processed! (loaded from cache)</h5></div>';
            } else {
                html += '<div class="alert alert-danger"><h5>❌ PDF processing failed!</h5></div>';
            }
        } else {
            html += `<div class="alert alert-primary"><h5>📦 Batch Processing Complete: ${total} PDF(s)</h5></div>`;
        }

        // Summary
        if (total > 1) {
            html += `
                <div class="card mb-3">
                    <div class="card-body">
                        <h6>📊 Summary:</h6>
                        <ul>
                            <li>✅ Newly processed: ${data.processed.length}</li>
                            <li>⊘ Cached (skipped): ${data.cached.length}</li>
                            <li>❌ Failed: ${data.failed.length}</li>
                        </ul>
                    </div>
                </div>
            `;
        }

        // Processed PDFs
        if (data.processed.length > 0) {
            html += '<div class="card mb-3"><div class="card-body">';
            html += '<h6>✅ Newly Processed:</h6>';
            data.processed.forEach(pdf => {
                html += `
                    <div class="mb-2">
                        <strong>📄 ${pdf.filename}</strong>
                        <ul class="small">
                            <li>Pages: ${pdf.num_pages}, Chunks: ${pdf.num_chunks}, Sections: ${pdf.num_sections}</li>
                            <li>Figures: ${pdf.num_figures}, Tables: ${pdf.num_tables}</li>
                            <li>Cross-refs: ${pdf.num_mentions} (${pdf.num_resolved_mentions} resolved)</li>
                        </ul>
                    </div>
                `;
            });
            html += '</div></div>';
        }

        // Cached PDFs
        if (data.cached.length > 0) {
            html += '<div class="card mb-3"><div class="card-body">';
            html += '<h6>⊘ Cached (Already Processed):</h6><ul>';
            data.cached.forEach(pdf => {
                html += `<li>${pdf.filename} (${pdf.num_chunks} chunks)</li>`;
            });
            html += '</ul></div></div>';
        }

        // Failed PDFs
        if (data.failed.length > 0) {
            html += '<div class="card border-danger mb-3"><div class="card-body">';
            html += '<h6 class="text-danger">❌ Failed:</h6><ul>';
            data.failed.forEach(pdf => {
                html += `<li>${pdf.filename}: ${pdf.error}</li>`;
            });
            html += '</ul></div></div>';
        }

        if (data.processed.length > 0 || data.cached.length > 0) {
            html += '<div class="alert alert-info">🤖 Ready to answer questions! Select a PDF from the Q&A tab.</div>';
        }

        return html;
    }

    function enableChat() {
        // Enable chat inputs if they exist
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');

        if (chatInput) chatInput.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
    }
});
