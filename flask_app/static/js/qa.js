/**
 * JavaScript for Q&A tab with Socket.IO chat
 */

document.addEventListener('DOMContentLoaded', () => {
    const pdfSelector = document.getElementById('pdfSelector');
    const refreshPdfBtn = document.getElementById('refreshPdfBtn');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const chatMessages = document.getElementById('chatMessages');
    const topKSlider = document.getElementById('topK');
    const topKValue = document.getElementById('topKValue');

    let chatHistory = [];

    // Update top K display
    if (topKSlider && topKValue) {
        topKSlider.addEventListener('input', () => {
            topKValue.textContent = topKSlider.value;
        });
    }

    // Enable/disable chat based on PDF selection
    if (pdfSelector) {
        pdfSelector.addEventListener('change', () => {
            const isSelected = pdfSelector.value !== '';
            if (chatInput) chatInput.disabled = !isSelected;
            if (sendBtn) sendBtn.disabled = !isSelected;

            if (isSelected) {
                chatInput.placeholder = 'Ask a question...';
            } else {
                chatInput.placeholder = 'Select a PDF first...';
            }
        });

        // Initialize state
        const isSelected = pdfSelector.value !== '';
        if (chatInput) chatInput.disabled = !isSelected;
        if (sendBtn) sendBtn.disabled = !isSelected;
    }

    // Refresh PDF list
    if (refreshPdfBtn) {
        refreshPdfBtn.addEventListener('click', async () => {
            refreshPdfBtn.disabled = true;
            refreshPdfBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            try {
                await window.pdfkg.refreshPdfList();
                window.pdfkg.showNotification('PDF list refreshed', 'success');
            } catch (error) {
                console.error('Refresh error:', error);
            } finally {
                refreshPdfBtn.disabled = false;
                refreshPdfBtn.innerHTML = '<i class="fas fa-sync"></i>';
            }
        });
    }

    // Send message
    function sendMessage() {
        const question = chatInput.value.trim();
        const pdfSlug = pdfSelector.value;

        if (!question || !pdfSlug) {
            return;
        }

        // Get LLM provider
        const llmProvider = document.querySelector('input[name="llmProvider"]:checked').value;
        const topK = parseInt(topKSlider.value);
        const embedModel = 'sentence-transformers/all-MiniLM-L6-v2';

        // Add user message to chat
        addChatMessage('user', question);

        // Clear input
        chatInput.value = '';

        // Disable input while processing
        chatInput.disabled = true;
        sendBtn.disabled = true;

        // Show thinking indicator
        showThinking();

        // Send message via Socket.IO
        window.pdfkg.socket.emit('chat_message', {
            question,
            pdf_slug: pdfSlug,
            llm_provider: llmProvider,
            top_k: topK,
            embed_model: embedModel
        });
    }

    // Send button click
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    // Enter key to send
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // Clear chat
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', () => {
            chatHistory = [];
            renderChat();
            window.pdfkg.showNotification('Chat history cleared', 'info');
        });
    }

    // Socket.IO event handlers for chat
    window.pdfkg.socket.on('chat_thinking', (data) => {
        console.log('Assistant is thinking...');
    });

    window.pdfkg.socket.on('chat_response', (data) => {
        console.log('Received chat response:', data);

        // Remove thinking indicator
        hideThinking();

        // Add assistant message to chat
        addChatMessage('assistant', data.answer, data.sources, data.debug);

        // Re-enable input
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    });

    window.pdfkg.socket.on('chat_error', (data) => {
        console.error('Chat error:', data);

        // Remove thinking indicator
        hideThinking();

        // Show error message
        addChatMessage('assistant', `❌ Error: ${data.error}`, null, null, true);

        // Re-enable input
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();

        window.pdfkg.showNotification(`Chat error: ${data.error}`, 'danger');
    });

    // Chat UI functions
    function addChatMessage(role, content, sources = null, debug = null, isError = false) {
        chatHistory.push({
            role,
            content,
            sources,
            debug,
            isError,
            timestamp: new Date()
        });
        renderChat();
    }

    function renderChat() {
        if (!chatMessages) return;

        // Clear existing messages except the example questions
        const exampleAlert = chatMessages.querySelector('.alert-info');
        chatMessages.innerHTML = '';

        // Re-add example questions if no messages yet
        if (chatHistory.length === 0 && exampleAlert) {
            chatMessages.appendChild(exampleAlert.cloneNode(true));
            return;
        }

        // Render all messages
        chatHistory.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${msg.role} fade-in`;

            let headerIcon = msg.role === 'user' ? '👤' : '🤖';
            let headerText = msg.role === 'user' ? 'You' : 'Assistant';

            let html = `
                <div class="message-header">
                    ${headerIcon} ${headerText}
                </div>
                <div class="message-content markdown-body">
                    ${window.pdfkg.formatMarkdown(msg.content)}
                </div>
            `;

            // Add sources if available
            if (msg.sources && msg.sources.length > 0) {
                html += '<div class="message-sources">';
                html += '<h6>📚 Sources:</h6>';
                html += '<ol>';
                msg.sources.forEach(source => {
                    html += `<li>${source.text}</li>`;
                });
                html += '</ol>';
                html += '</div>';
            }

            messageDiv.innerHTML = html;
            chatMessages.appendChild(messageDiv);
        });

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showThinking() {
        const thinkingDiv = document.createElement('div');
        thinkingDiv.id = 'thinkingIndicator';
        thinkingDiv.className = 'chat-thinking';
        thinkingDiv.innerHTML = `
            <div class="spinner-border spinner-border-sm" role="status">
                <span class="visually-hidden">Thinking...</span>
            </div>
            Thinking...
        `;
        chatMessages.appendChild(thinkingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideThinking() {
        const thinkingDiv = document.getElementById('thinkingIndicator');
        if (thinkingDiv) {
            thinkingDiv.remove();
        }
    }

    // Load initial PDF list
    window.pdfkg.refreshPdfList();
});
