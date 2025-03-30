# wsgi.py
from app import create_app
app = create_app('production')  # Use production config

if __name__ == "__main__":
    app.run()
