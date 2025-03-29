# models/job_description.py
"""
Job Description Model - Stores and manages job description data.
Includes deduplication via content hashing.
"""

from datetime import datetime, timezone
from extensions import db
from sqlalchemy.dialects.postgresql import JSONB, JSON

class JobDescription(db.Model):
    __tablename__ = 'job_descriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    raw_text = db.Column(db.Text, nullable=False)
    # Mixed JSON type: use JSONB for PostgreSQL and JSON for SQLite.
    analysis_data = db.Column(
        JSONB().with_variant(JSON(), 'sqlite'),
        nullable=False
    )
    # Content hash for deduplication (SHA-256 hash of raw_text).
    content_hash = db.Column(db.String(64), unique=True, index=True, nullable=False)
    
    # Optional: record creation time.
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    analyses = db.relationship('AnalysisResult', back_populates='job_description', lazy='dynamic')

    def __repr__(self):
        """Return a string representation of the job description model,
        displaying the first 8 characters of the content hash.
        """
        return f'<JobDescription {self.content_hash[:8] if self.content_hash else "None"}>'

    def calculate_hash(self) -> str:
        """Generate SHA-256 hash of the raw_text content.

        Returns:
            str: The hexadecimal SHA-256 hash of the raw_text.
        """
        import hashlib
        return hashlib.sha256(self.raw_text.encode('utf-8')).hexdigest()

    def before_save(self):
        """Automatically calculate and set the content_hash before saving,
        if it has not been set already.
        """
        if not self.content_hash:
            self.content_hash = self.calculate_hash()
