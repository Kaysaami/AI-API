# models/upload.py
"""
Upload Model - Database model for tracking uploaded resumes/CVs.
Stores metadata about the file upload along with processed text and
establishes relationships with the user and analysis result.
"""

from datetime import datetime, timezone
from extensions import db

class Upload(db.Model):
    """Database model for tracking uploaded resumes/CVs."""
    __tablename__ = 'uploads'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    processed_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships (using string references to avoid circular imports)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    user = db.relationship('User', back_populates='uploads')
    
    # One-to-one relationship with AnalysisResult
    analysis = db.relationship('AnalysisResult', back_populates='upload', uselist=False)

    def __repr__(self):
        """Return a string representation of the Upload model."""
        return f'<Upload {self.filename}>'
