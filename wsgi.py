# wsgi.py
from app import create_app  # Adjust 'app' to your main file’s name if different

app = create_app()  # Assuming create_app() is your factory function

if __name__ == "__main__":
    app.run()  # Optional, for local testing
