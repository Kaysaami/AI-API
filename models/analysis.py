# models/ analysis.py
"""
Analysis Result Model - Stores resume analysis outcomes.
Connects users, uploads, and job descriptions.
"""

from datetime import datetime, timezone
from extensions import db
from sqlalchemy.dialects.postgresql import JSONB, JSON


class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'

    id = db.Column(db.Integer, primary_key=True)
    # Use JSONB for PostgreSQL and JSON for SQLite
    analysis_data = db.Column(
        JSONB().with_variant(JSON(), 'sqlite'),
        nullable=False
    )
    resume_text = db.Column(db.Text, nullable=False)
    job_description_text = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float, nullable=False)
    industry = db.Column(db.String(50), nullable=True)
    analysis_version = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Foreign Keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), index=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('uploads.id'), index=True)
    job_description_id = db.Column(db.Integer, db.ForeignKey('job_descriptions.id'), index=True)

    # Relationships
    user = db.relationship('User', back_populates='analyses')
    upload = db.relationship('Upload', back_populates='analysis')
    job_description = db.relationship('JobDescription', back_populates='analyses')

    __table_args__ = (
        db.CheckConstraint('score >= 0 AND score <= 1', name='valid_score_range'),
    )

    def __repr__(self):
        """Return a string representation of the analysis result."""
        return f'<Analysis {self.id} ({self.score:.0%})>'
