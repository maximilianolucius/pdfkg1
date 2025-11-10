"""Main routes for web pages."""
from flask import Blueprint, render_template, current_app, redirect, url_for
from flask_app.system_logger import system_logger
from pdfkg.template_extractor import available_submodels
from pdfkg.submodel_templates import get_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Landing page with tabs navigation."""
    storage = current_app.config['storage']

    # Get available PDFs
    pdf_list = storage.list_pdfs()

    # Get submodels for AASX tab
    submodels = available_submodels()
    submodel_choices = [(get_template(key).display_name, key) for key in submodels]

    return render_template('index.html',
                         pdf_list=pdf_list,
                         submodel_choices=submodel_choices)


@main_bp.route('/ingest')
def ingest():
    """PDF ingestion page."""
    return render_template('ingest.html')


@main_bp.route('/qa')
def qa():
    """Q&A interface page."""
    storage = current_app.config['storage']
    pdf_list = storage.list_pdfs()

    return render_template('qa.html', pdf_list=pdf_list)


@main_bp.route('/aasx')
def aasx():
    """AASX generation page."""
    submodels = available_submodels()
    submodel_choices = [(get_template(key).display_name, key) for key in submodels]

    return render_template('aasx.html', submodel_choices=submodel_choices)


@main_bp.route('/logs')
def logs():
    """System logs page."""
    return render_template('logs.html')


@main_bp.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'healthy', 'service': 'pdfkg-flask'}, 200
