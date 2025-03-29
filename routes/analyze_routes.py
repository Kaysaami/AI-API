#routes.analyze_routes.py
"""
Analyze Routes - Endpoints for resume and job description analysis.
Provides endpoints for full resume analysis and job matching analysis.
"""

import logging
from flask import Blueprint, request, jsonify
from services.resume_analyzer import analyzer
from services.feedback_generator import ResumeFeedbackGenerator

analyze_bp = Blueprint('analyze', __name__)

@analyze_bp.route('/analyze', methods=['POST'])
def analyze_resume():
    """
    Endpoint for analyzing a resume with an optional job description.
    Expected JSON payload (accepts both formats):
    {
        "resume_text": "...", (or "full_text": "...")
        "job_description": "..."  (optional),
        "sections": {...}  (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON data"}), 400
            
        # Handle both "resume_text" and "full_text" keys for flexibility
        resume_text = data.get('resume_text') or data.get('full_text')
        if not resume_text:
            return jsonify({"error": "Missing resume text. Please provide 'resume_text' or 'full_text'"}), 400

        job_description = data.get('job_description')
        sections = data.get('sections')
        
        # Basic validation for sections structure
        if sections is not None and not isinstance(sections, dict):
            return jsonify({"error": "Invalid 'sections' format. Expected a dictionary"}), 400
            
        # Use SectionExtractor to properly validate sections if provided
        if sections:
            try:
                # Import SectionExtractor if not already available
                from services.section_extractor import SectionExtractor
                section_extractor = SectionExtractor()
                
                # Validate sections using the SectionExtractor method
                is_valid, normalized_sections, warning_msg = section_extractor.validate_sections(sections)
                
                if not is_valid:
                    return jsonify({"error": f"Invalid sections structure: {warning_msg}"}), 400
                    
                # Use the normalized sections
                sections = normalized_sections
                
                if warning_msg:
                    logging.warning(warning_msg)
                    
            except Exception as e:
                logging.error(f"Error validating sections: {str(e)}")
                # Don't fail the request, just log the error and continue with original sections

        # Process the analysis
        analysis_result = analyzer.analyze_resume(resume_text, job_description, sections=sections)

        # Generate feedback
        feedback_generator = ResumeFeedbackGenerator()
        feedback = feedback_generator.generate_feedback(
            analysis_result, 
            job_description=job_description
        )

        # Prepare response with analysis and feedback
        response = {
            "analysis": {
                "basic_stats": analysis_result.get('basic_stats', {}),
                "skills_analysis": analysis_result.get('skills', {}),
                "experience_analysis": analysis_result.get('experience', {}),
                "education_analysis": analysis_result.get('education', {})
            },
            "feedback": feedback
        }
        
        # Add job match data if available
        if job_description and 'job_match' in analysis_result:
            response["job_match"] = {
                "match_score": f"{analysis_result['job_match']['overall_score'] * 100:.1f}%",
                "skills_match": analysis_result['job_match']['skills_match'],
                "experience_match": analysis_result['job_match']['experience_match'],
                "requirements_match": analysis_result['job_match']['requirements_match'],
                "industry": analysis_result['job_match']['industry']
            }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

@analyze_bp.route('/analyze/match', methods=['POST'])
def analyze_job_match():
    """
    Endpoint specifically for job matching analysis.
    Expected JSON payload:
    {
        "resume_text": "...",
        "job_description": "..."
    }
    """
    try:
        data = request.get_json()

        # Validate input
        required_fields = ['resume_text', 'job_description']
        if not data or not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing required fields",
                "required_fields": required_fields
            }), 400

        # Perform analysis
        analysis_result = analyzer.analyze_resume(
            data['resume_text'],
            data['job_description']
        )

        # Extract just the job match information
        if 'job_match' in analysis_result:
            response = {
                "match_score": f"{analysis_result['job_match']['overall_score'] * 100:.1f}%",
                "skills_match": analysis_result['job_match']['skills_match'],
                "experience_match": analysis_result['job_match']['experience_match'],
                "requirements_match": analysis_result['job_match']['requirements_match']
            }
            return jsonify(response)
        else:
            return jsonify({"error": "Job matching analysis failed"}), 500

    except Exception as e:
        logging.error(f"Job matching analysis failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Job matching analysis failed",
            "details": str(e)
        }), 500
