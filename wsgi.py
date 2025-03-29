# wsgi.py
from app import create_app

app = create_app('production')  # Adjust config name as needed