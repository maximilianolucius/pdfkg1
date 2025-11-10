"""Socket.IO event handlers for real-time communication."""
from flask import request
from flask_socketio import emit, join_room, leave_room
from flask_app.system_logger import system_logger
from pdfkg.query import answer_question
import json


def register_handlers(socketio, storage):
    """Register Socket.IO event handlers."""

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        sid = request.sid
        system_logger.log(f"Client connected: {sid}", "INFO")
        emit('connected', {'message': 'Connected to PDFKG server', 'sid': sid})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        sid = request.sid
        system_logger.log(f"Client disconnected: {sid}", "INFO")

    @socketio.on('join_logs')
    def handle_join_logs():
        """Join the logs room for real-time log updates."""
        join_room('logs')
        system_logger.log(f"Client {request.sid} joined logs room", "INFO")
        emit('joined_logs', {'message': 'Joined logs room'})

    @socketio.on('leave_logs')
    def handle_leave_logs():
        """Leave the logs room."""
        leave_room('logs')
        system_logger.log(f"Client {request.sid} left logs room", "INFO")
        emit('left_logs', {'message': 'Left logs room'})

    @socketio.on('get_logs')
    def handle_get_logs():
        """Get current system logs."""
        try:
            logs = system_logger.get_logs_list()
            emit('logs_update', {'logs': logs})
        except Exception as e:
            emit('error', {'message': f'Error getting logs: {str(e)}'})

    @socketio.on('chat_message')
    def handle_chat_message(data):
        """Handle incoming chat message and respond with answer."""
        try:
            question = data.get('question')
            pdf_slug = data.get('pdf_slug')
            llm_provider = data.get('llm_provider', 'gemini')
            top_k = data.get('top_k', 5)
            embed_model = data.get('embed_model', 'sentence-transformers/all-MiniLM-L6-v2')

            if not question or not pdf_slug:
                emit('chat_error', {'error': 'Missing question or PDF selection'})
                return

            system_logger.log(f"Chat query: '{question}' on PDF: {pdf_slug}", "INFO")

            # Emit thinking status
            emit('chat_thinking', {'message': 'Thinking...'})

            # Get answer
            result = answer_question(
                question=question,
                pdf_slug=pdf_slug,
                model_name=embed_model,
                top_k=top_k,
                llm_provider=llm_provider,
                storage=storage,
            )

            answer = result.get("answer", "No answer could be generated.")
            sources = result.get("sources", [])

            # Format source references
            source_refs = []
            if sources:
                for i, src in enumerate(sources[:5], 1):
                    if 'text' in src:  # It's a chunk
                        page = src.get('page', 'N/A')
                        section = src.get('section_id', 'N/A')
                        score = src.get('similarity_score')
                        score_str = f"(score: {score:.2f})" if score else ""
                        source_refs.append({
                            'index': i,
                            'text': f"Section {section}, Page {page} {score_str}",
                            'type': 'chunk',
                            'page': page,
                            'section': section,
                            'score': score
                        })
                    else:  # It's a graph node
                        node_type = src.get('type', 'Node')
                        label = src.get('label', src.get('_key', 'N/A'))
                        source_refs.append({
                            'index': i,
                            'text': f"{node_type}: {label}",
                            'type': 'node',
                            'node_type': node_type,
                            'label': label
                        })

            # Emit response
            emit('chat_response', {
                'question': question,
                'answer': answer,
                'sources': source_refs,
                'debug': result.get('debug', {})
            })

            system_logger.log(f"Chat response sent for: '{question}'", "INFO")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            system_logger.log(f"Chat error: {error_details}", "ERROR")
            emit('chat_error', {
                'error': str(e),
                'details': error_details
            })

    @socketio.on('processing_progress')
    def handle_processing_progress(data):
        """Handle progress updates during PDF processing."""
        # This can be used by clients to emit progress updates
        # which will be broadcast to all connected clients
        emit('progress_update', data, broadcast=True)

    # Function to broadcast log updates (can be called from other modules)
    def broadcast_log(log_entry):
        """Broadcast log entry to all clients in logs room."""
        socketio.emit('new_log', {'log': log_entry}, room='logs')

    # Attach broadcast function to system logger (optional enhancement)
    # This would require modifying system_logger to accept a callback
    # For now, clients will poll or request logs manually

    return socketio
