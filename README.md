# AI-Powered Resume Analyzer & Matcher

A Flask-based REST API designed to analyze resumes, match them against job descriptions, and provide actionable feedback for resume optimization using ruled-based, AI and NLP techniques.

## Project Overview

Moth Byte’s AI-Powered Resume Analyzer combines Natural Language Processing (NLP) and machine learning to transform resume optimization. Users upload their resume (PDF, DOCX, or text) and optionally a target job description. The system performs a detailed analysis, including:

* **Structural Validation:** Checks formatting compliance, section order, and keyword placement using text layout parsing and pattern matching.
* **Content Optimization:** Leverges NLP (spaCy) to extract skills, experience, and education, comparing them against job descriptions to identify gaps and suggest improvements.
* **ATS Simulation:** Generates a compatibility score and provides actionable feedback with targeted suggestions to optimize the resume for Applicant Tracking Systems.

## Architecture

The project follows a two-repository structure:

1.  **`resume-matcher-core`:** Contains the core business logic, NLP models, parsing utilities, and analysis services. This library is packaged and used as a dependency by the API.
2.  **`resume-matcher-api`:** A Flask-based web application providing a REST API for interacting with the core services. It handles user authentication, file uploads, database interactions, and API routing.

## Key Features (API)

* **Resume/CV Processing:** Upload and extract text from PDF and DOCX files.
* **Resume Analysis:** Extract skills, experience, education, and other key information using the core library.
* **Job Matching:** Compare resumes to job descriptions and calculate match scores.
* **User Authentication:** Secure JWT-based authentication with password hashing.
* **Role-Based Access Control:** Admin and user roles with distinct permissions.
* **RESTful API Design:** Well-structured API endpoints with clear request/response formats and error handling.
* **Database Management:** Uses SQLAlchemy ORM for data persistence and Flask-Migrate for schema migrations.

## Key Innovations

* **Dual Analysis:** Combines content analysis (NLP entity/section detection) with document structure validation (text-based layout parsing) through unified text processing.
* **Explainable AI:** Highlights exact phrases needing revision instead of generic advice.
* **Privacy-First Design:** Local multi-format processing (PDF/DOCX/text) with no third-party data sharing (within the core library).

## Tech Stack

* **Backend:**
    * Flask: Lightweight web framework for the API.
    * SQLAlchemy: ORM for database interactions.
    * Flask-Migrate: Manages database migrations.
    * Flask-CORS: Enables cross-origin resource sharing.
    * PyJWT: Implements JWT authentication.
    * Celery: Handles asynchronous processing tasks.
    * Gunicorn: WSGI server for deployment.
    * python-dotenv: Manages environment variables.
* **AI/NLP (Core Library):**
    * spaCy: For entity recognition, section parsing, and core NLP tasks (Transformer models).
    * NLTK: Used for WordNet (synonyms).
    * fuzzywuzzy: For fuzzy string matching.
    * scikit-learn: For TF-IDF keyword scoring and similarity analysis.
    * Transformers (Hugging Face): Optional integration for advanced NLP (e.g., SentenceTransformers for embeddings, zero-shot classification).
    * OpenAI API: For LLM-based dynamic feedback generation.
* **File Parsing (Core Library):**
    * python-docx: For `.docx` files.
    * PyPDF2 / pdfplumber: For `.pdf` files.
* **Frontend:**
    * React.js: JavaScript library for building user interfaces.
    * Material-UI: React UI component library.
    * Chart.js: For data visualizations.
* **Database:**
    * PostgreSQL: Primary database for production and development.
    * SQLite: Used for initial local development/testing.
* **Infrastructure & Packaging:**
    * setup.py: For packaging the `resume-matcher-core` library.
    * Procfile / runtime.txt: For deployments.
    * Modular architecture suitable for cloud deployment (AWS, GCP, Azure).

## Project Structure

```env
resume-matcher-core/
├── resume_matcher_services/
│   ├── __init__.py           # Exports core services
│   ├── security.py           # Authentication and security utilities
│   ├── database_utils.py     # Database session management
│   ├── shared_utilities.py   # Common utilities (regex, industry terms)
│   ├── file_parser.py        # PDF/DOCX parsing
│   ├── skills_extractor.py   # NLP-based skill extraction
│   ├── industry_config.py    # Industry-specific configurations
│   ├── job_analyzer.py       # Job description parsing
│   ├── education_parser.py   # Education section parsing
│   ├── experience_parser.py  # Experience section parsing
│   ├── resume_analyzer.py    # Core resume analysis logic
│   ├── section_extractor.py  # Section detection
│   ├── text_analysis.py      # Text similarity and NLP processing
│   └── feedback_generator.py # Feedback generation
├── .github/workflows/
│   └── publish.yml           # CI/CD pipeline
├── setup.py                  # Package setup
├── README.md                 # Core documentation
├── requirements.txt          # Dependencies
└── .gitignore                # Ignored files

resume-matcher-api/
├── app.py                    # Flask app entry point
├── config.py                 # Configuration settings
├── error_handlers.py         # Custom error handling
├── extensions.py             # Flask extensions setup
├── models/
│   ├── __init__.py           # Model exports
│   ├── user.py               # User model
│   ├── analysis.py           # Analysis model
│   ├── job_description.py    # Job description model
│   ├── upload.py             # Upload tracking model
├── services/
│   ├── __init__.py           # Service imports
│   ├── security_db.py        # Authentication services
├── routes/
│   ├── __init__.py           # Route setup
│   ├── auth_routes.py        # Auth endpoints
│   ├── analyze_routes.py     # Analysis endpoints
│   ├── upload_routes.py      # Upload endpoints
│   ├── job_routes.py         # Job description endpoints
│   ├── admin_routes.py       # Admin endpoints
│   ├── auth_utils.py         # Auth utilities
├── migrations/               # Database migrations
├── uploads/                  # File storage (gitignored)
├── requirements.txt          # Dependencies
├── Procfile                  # Gunicorn setup
├── runtime.txt               # Python version (3.12.6)
├── wsgi.py                   # WSGI entry point
├── .gitignore                # Ignored files
├── .env                      # Environment variables
└── README.md                 # API documentation

```
## Getting Started (API)

These instructions guide you through setting up and running the `resume-matcher-api`.

### Prerequisites

* Python 3.10+ (the best tested version for this project is 3.12.6)
* pip (Python package manager)
* Virtual environment tool (e.g., `venv`)
* PostgreSQL Server running
* Access to the `resume_matcher_services` wheel file (`.whl`) built from the private `resume-matcher-core` repository.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kaysaami/resume-matcher-api.git
    cd resume-matcher-api
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create venv
    python -m venv venv

    # Activate venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the core library:**
    Place the `resume_matcher_services-*.whl` file in the project root or provide the correct path.
    ```bash
    pip install path/to/resume_matcher_services-0.1.0-py3-none-any.whl
    ```
    *(Replace `path/to/` and the version number as needed)*

4.  **Install API dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    Make sure `.env` file in the root directory. 
   
    **Important:** Ensure your PostgreSQL server is running and the specified database (`resume_matcher_dev`) exists. You might need to create the user and database manually.
     ```env
    # PostgreSQL default values
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=MothByte2025
    POSTGRES_DB=resume_matcher_dev, resume_matcher_test
    DATABASE_URL=postgresql://postgres:MothByte2025@localhost:5432/resume_matcher_dev
    
    ```
6.  **Database Migrations:**
    Initialize and apply database migrations:
    ```bash
    flask db init # Only needed the first time
    flask db migrate -m "Initial migration"
    flask db upgrade
    ```

7.  **Run the Application:**
    ```bash
    python app.py
    ```
    Or using Gunicorn (Only for deployment):
    ```bash
    gunicorn wsgi:app
    ```
    If using Celery, start a worker separately:
    ```bash
    celery -A your_celery_app_instance worker --loglevel=info
    ```
    The server will typically start on `http://127.0.0.1:5000`.

### Test Users

* The first user to register in a fresh database automatically becomes an admin
* Subsequent users are regular users by default
* Admin privileges for additional users must be granted by an existing admin

## API Endpoints

*(Based on the route definitions in `routes/`)*

### Authentication (`/auth`)

* `POST /auth/register`: Register a new user.
* `POST /auth/login`: Log in user, returns JWT token.
* `GET /auth/me`: Get current authenticated user's profile (`@token_required`) for authenticated users.
* `POST /auth/password-reset-request`: Initiate password reset process.
* `POST /auth/password-reset`: Complete password reset using a token.
* `POST /auth/change-password`: Change password for the authenticated user (`@token_required`).

### File Upload (`/`)

* `POST /upload`: Upload a resume/CV file (PDF/DOCX). Returns extracted text and sections.

### Analysis (`/`)

* `POST /analyze`: Analyze resume text (from upload or direct input) against an optional job description (text or ID). Returns analysis and feedback.
* `GET /analyze/<int:analysis_id>`: Retrieve a specific analysis result for authenticated users.

### Job Descriptions (`/job`)

* `POST /job/text`: Submit job description text directly. Stores it and returns ID.
* `GET /job/<int:job_id>`: Retrieve a specific job description by its ID.
* `PUT /job/<int:job_id>`: Update a job description for authenticated users
* `DELETE /job/<int:job_id>`: Delete a job description for authenticated users

### Admin Functions (`/admin`)

*(All endpoints require admin privileges via `@admin_required`)*

* `GET /admin/users`: List all registered users.
* `GET /admin/users/<int:user_id>`: Get details for a specific user.
* `POST /admin/users/<int:user_id>/toggle-status`: Activate or deactivate a user account.
* `POST /admin/users/<int:user_id>/toggle-admin`: Grant or revoke admin privileges for a user.
* `GET /admin/stats`: Retrieve system statistics (e.g., user counts).

### Utility (`/`)

* `GET /health` - API health check
* `GET /` - API documentation overview
