from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from extensions import db, init_extensions
from config import Config

def create_app(config_name='development'):
    app = Flask(__name__)
    
    # Load configuration based on environment
    config_settings = Config.get_config()[config_name]
    app.config.update(config_settings)
    
    # Initialize extensions
    init_extensions(app)
    
    # Register blueprints
    from routes.upload_routes import upload_bp
    from routes.analyze_routes import analyze_bp
    from routes.auth_routes import auth_bp
    from routes.admin_routes import admin_bp

    app.register_blueprint(upload_bp)
    app.register_blueprint(analyze_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    
    # Import all model modules to register them with SQLAlchemy in the right order
    with app.app_context():
        # Import models in the correct order
        from models.user import User
        from models.upload import Upload
        from models.analysis import AnalysisResult
        
        # Create tables
        db.create_all()
        
    # Register error handlers
    from error_handlers import register_error_handlers
    register_error_handlers(app)
    
    return app

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))  # Use DigitalOcean's PORT, default to 8080
    app.run(host="0.0.0.0", port=port)
