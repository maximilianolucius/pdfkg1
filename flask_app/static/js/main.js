/**
 * Main JavaScript for PDFKG Flask App
 * Handles Socket.IO connection and common utilities
 */

// Initialize Socket.IO connection
const socket = io({
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5
});

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('✅ Connected to server via Socket.IO');
    showNotification('Connected to server', 'success');
});

socket.on('disconnect', () => {
    console.log('❌ Disconnected from server');
    showNotification('Disconnected from server', 'warning');
});

socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
    showNotification('Connection error. Retrying...', 'danger');
});

socket.on('error', (data) => {
    console.error('Socket error:', data);
    showNotification(data.message || 'An error occurred', 'danger');
});

// Utility functions
function showNotification(message, type = 'info', duration = 3000) {
    /**
     * Show a Bootstrap toast notification
     */
    const toastContainer = document.getElementById('toastContainer');

    if (!toastContainer) {
        // Create toast container if it doesn't exist
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    document.getElementById('toastContainer').appendChild(toast);

    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: duration
    });

    bsToast.show();

    // Remove toast element after it's hidden
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

function showSpinner(message = 'Loading...') {
    /**
     * Show loading spinner overlay
     */
    const spinner = document.createElement('div');
    spinner.id = 'loadingSpinner';
    spinner.className = 'spinner-overlay';
    spinner.innerHTML = `
        <div class="text-center text-white">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">${message}</p>
        </div>
    `;
    document.body.appendChild(spinner);
}

function hideSpinner() {
    /**
     * Hide loading spinner overlay
     */
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.remove();
    }
}

function formatMarkdown(text) {
    /**
     * Convert markdown text to HTML using marked.js
     */
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    return text.replace(/\n/g, '<br>');
}

function apiRequest(endpoint, options = {}) {
    /**
     * Make API request with error handling
     */
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json'
        }
    };

    const mergedOptions = { ...defaultOptions, ...options };

    return fetch(`/api${endpoint}`, mergedOptions)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .catch(error => {
            console.error('API request error:', error);
            showNotification(`API Error: ${error.message}`, 'danger');
            throw error;
        });
}

function refreshPdfList() {
    /**
     * Refresh the PDF selector dropdown
     */
    return apiRequest('/pdfs')
        .then(data => {
            if (data.success) {
                const selector = document.getElementById('pdfSelector');
                if (selector) {
                    // Save current selection
                    const currentValue = selector.value;

                    // Clear and rebuild options
                    selector.innerHTML = '<option value="">Select a PDF...</option>';

                    data.pdfs.forEach(pdf => {
                        const option = document.createElement('option');
                        option.value = pdf.slug;
                        option.textContent = `${pdf.filename} (${pdf.num_chunks} chunks)`;
                        selector.appendChild(option);
                    });

                    // Restore selection if still valid
                    if (currentValue && data.pdfs.some(p => p.slug === currentValue)) {
                        selector.value = currentValue;
                    }
                    // Auto-select if only one PDF and no current selection
                    else if (!currentValue && data.pdfs.length === 1) {
                        selector.value = data.pdfs[0].slug;
                    }

                    // Trigger change event to update UI
                    selector.dispatchEvent(new Event('change'));
                }
                return data.pdfs;
            }
        });
}

// Tab change handling
document.addEventListener('DOMContentLoaded', () => {
    // Handle tab changes
    const tabButtons = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabButtons.forEach(button => {
        button.addEventListener('shown.bs.tab', (event) => {
            const targetTab = event.target.dataset.bsTarget;
            console.log(`Switched to tab: ${targetTab}`);

            // Trigger tab-specific initialization
            if (targetTab === '#logs') {
                // Join logs room when switching to logs tab
                if (typeof joinLogsRoom === 'function') {
                    joinLogsRoom();
                }
            }
        });

        button.addEventListener('hidden.bs.tab', (event) => {
            const targetTab = event.target.dataset.bsTarget;

            // Cleanup tab-specific resources
            if (targetTab === '#logs') {
                // Leave logs room when switching away from logs tab
                if (typeof leaveLogsRoom === 'function') {
                    leaveLogsRoom();
                }
            }
        });
    });

    // Handle hash navigation
    const hash = window.location.hash;
    if (hash) {
        const tabButton = document.querySelector(`[data-bs-target="${hash}"]`);
        if (tabButton) {
            const tab = new bootstrap.Tab(tabButton);
            tab.show();
        }
    }

    // Update hash when tab changes
    tabButtons.forEach(button => {
        button.addEventListener('shown.bs.tab', (event) => {
            const targetTab = event.target.dataset.bsTarget;
            window.location.hash = targetTab;
        });
    });

    console.log('PDFKG Flask App initialized');
});

// Export functions for use in other scripts
window.pdfkg = {
    socket,
    showNotification,
    showSpinner,
    hideSpinner,
    formatMarkdown,
    apiRequest,
    refreshPdfList
};
