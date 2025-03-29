# routes/upload_routes.py
"""
Upload Routes - Endpoint for processing and storing resume/CV uploads.
Handles file validation, text extraction, and section detection.
"""

import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from extensions import db, limiter
from services.file_parser import extract_text
from services.section_extractor import SectionExtractor
from models.upload import Upload

upload_bp = Blueprint('upload', __name__)
section_extractor = SectionExtractor()

@upload_bp.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    """
    Endpoint to upload a resume/CV file.
    Processes the file to extract text and detect sections, then creates
    a database record for the upload.
    """
    try:
        # Log request information
        logging.info("=== Upload Request Details ===")
        logging.info(f"Content-Type: {request.content_type}")
        logging.info(f"Content-Length: {request.content_length}")
        logging.info(f"Files in request: {list(request.files.keys())}")

        if 'file' not in request.files:
            logging.error("No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if not file.filename:
            logging.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Log file details
        logging.info("=== File Details ===")
        logging.info(f"Filename: {file.filename}")
        logging.info(f"Content Type: {file.content_type}")
        logging.info(f"Headers: {file.headers}")

        # Process the file
        try:
            result = extract_text(file)
            
            # Validate that we got text back
            if not result.get('full_text'):
                return jsonify({"error": "Failed to extract text from file"}), 400

            # Extract sections from the full text
            try:
                sections = section_extractor.detect_sections(result['full_text'])
                
                # Validate the sections using SectionExtractor's method
                is_valid, normalized_sections, warning_msg = section_extractor.validate_sections(sections)
                
                if not is_valid:
                    logging.error(f"Invalid sections: {warning_msg}")
                    # Fall back to empty dict if invalid
                    sections = {}
                else:
                    sections = normalized_sections
                    if warning_msg:  # This is just a warning
                        logging.warning(warning_msg)
                        
            except Exception as e:
                logging.error(f"Section extraction failed: {str(e)}")
                sections = {}  # Fallback to empty dict

            # Create and commit the upload record in the database
            new_upload = Upload(
                filename=file.filename,
                processed_text=result['full_text']
            )
            db.session.add(new_upload)
            db.session.commit()

            # Prepare the response with data that will be compatible with the analyze route
            return jsonify({
                "message": "File processed successfully",
                "filename": file.filename,
                "text_preview": result['full_text'][:200] + "..." if len(result['full_text']) > 200 else result['full_text'],
                "full_text": result['full_text'],
                "resume_text": result['full_text'],  # Added for direct compatibility with analyze route
                "sections": sections,
                "upload_id": new_upload.id
            })

        except ValueError as e:
            logging.error(f"Extraction error: {str(e)}")
            return jsonify({"error": str(e)}), 400

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500