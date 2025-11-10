/**
 * JavaScript for System Logs tab with Socket.IO
 */

let logsRefreshInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    const refreshLogsBtn = document.getElementById('refreshLogsBtn');
    const clearLogsBtn = document.getElementById('clearLogsBtn');
    const autoRefreshLogs = document.getElementById('autoRefreshLogs');
    const logsContent = document.getElementById('logsContent');

    // Join logs room when tab becomes active
    window.joinLogsRoom = () => {
        if (window.pdfkg && window.pdfkg.socket) {
            window.pdfkg.socket.emit('join_logs');
            console.log('Joined logs room');
            refreshLogs();
        }
    };

    // Leave logs room when tab becomes inactive
    window.leaveLogsRoom = () => {
        if (window.pdfkg && window.pdfkg.socket) {
            window.pdfkg.socket.emit('leave_logs');
            console.log('Left logs room');
        }
    };

    // Refresh logs
    async function refreshLogs() {
        try {
            const data = await window.pdfkg.apiRequest('/logs');
            if (data.success && data.logs) {
                displayLogs(data.logs);
            }
        } catch (error) {
            console.error('Error refreshing logs:', error);
        }
    }

    // Display logs
    function displayLogs(logs) {
        if (!logsContent) return;

        if (logs.length === 0) {
            logsContent.textContent = 'No logs yet.';
        } else {
            logsContent.textContent = logs.join('\n');
            // Auto-scroll to bottom
            logsContent.parentElement.scrollTop = logsContent.parentElement.scrollHeight;
        }
    }

    // Refresh button
    if (refreshLogsBtn) {
        refreshLogsBtn.addEventListener('click', () => {
            refreshLogsBtn.disabled = true;
            refreshLogsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            refreshLogs().finally(() => {
                refreshLogsBtn.disabled = false;
                refreshLogsBtn.innerHTML = '<i class="fas fa-sync"></i> Refresh Logs';
            });
        });
    }

    // Clear logs button
    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', async () => {
            if (!confirm('Clear all logs?')) {
                return;
            }

            clearLogsBtn.disabled = true;
            clearLogsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            try {
                const data = await window.pdfkg.apiRequest('/logs/clear', {
                    method: 'POST'
                });

                if (data.success) {
                    logsContent.textContent = 'Logs cleared.';
                    window.pdfkg.showNotification('Logs cleared', 'success');
                }
            } catch (error) {
                console.error('Error clearing logs:', error);
            } finally {
                clearLogsBtn.disabled = false;
                clearLogsBtn.innerHTML = '<i class="fas fa-trash"></i> Clear Logs';
            }
        });
    }

    // Auto-refresh toggle
    if (autoRefreshLogs) {
        autoRefreshLogs.addEventListener('change', () => {
            if (autoRefreshLogs.checked) {
                startAutoRefresh();
            } else {
                stopAutoRefresh();
            }
        });

        // Start auto-refresh if checked by default
        if (autoRefreshLogs.checked) {
            startAutoRefresh();
        }
    }

    function startAutoRefresh() {
        if (logsRefreshInterval) {
            clearInterval(logsRefreshInterval);
        }
        logsRefreshInterval = setInterval(refreshLogs, 5000); // Refresh every 5 seconds
        console.log('Auto-refresh started');
    }

    function stopAutoRefresh() {
        if (logsRefreshInterval) {
            clearInterval(logsRefreshInterval);
            logsRefreshInterval = null;
        }
        console.log('Auto-refresh stopped');
    }

    // Socket.IO event handlers for real-time logs
    if (window.pdfkg && window.pdfkg.socket) {
        window.pdfkg.socket.on('logs_update', (data) => {
            if (data.logs) {
                displayLogs(data.logs);
            }
        });

        window.pdfkg.socket.on('new_log', (data) => {
            // Append new log entry
            if (logsContent && data.log) {
                if (logsContent.textContent === 'No logs yet.') {
                    logsContent.textContent = data.log;
                } else {
                    logsContent.textContent += '\n' + data.log;
                }
                // Auto-scroll to bottom
                logsContent.parentElement.scrollTop = logsContent.parentElement.scrollHeight;
            }
        });

        window.pdfkg.socket.on('joined_logs', () => {
            console.log('Successfully joined logs room');
        });

        window.pdfkg.socket.on('left_logs', () => {
            console.log('Successfully left logs room');
        });
    }

    // Load initial logs
    refreshLogs();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.leaveLogsRoom) {
        window.leaveLogsRoom();
    }
});
