# services/shared_utilities.py
import re
import os
import yaml  # Import PyYAML for YAML parsing

def load_config():
    """
    Loads a configuration dictionary for section extraction.
    Returns a dict with keys:
      - "standard_sections"
      - "section_synonyms"
      - "section_patterns"
    It attempts to load an override from a YAML file (if available),
    and otherwise returns the default configuration built from constants.
    """
    default_config = {
        "standard_sections": STANDARD_SECTIONS,
        "section_synonyms": SECTION_SYNONYMS,
        "section_patterns": SECTION_PATTERNS_RAW
    }
    # Optionally, try to load a config file named "section_config.yaml" from the config directory.
    config_path = os.path.join(CONFIG_DIR, "section_config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Merge the loaded config with defaults
            default_config.update(config)
        except Exception as e:
            print(f"Error loading section config from {config_path}: {e}")
    return default_config
# -------------------------------
# External Configuration Loader
# -------------------------------
def load_external_config(filepath, default):
    """Attempt to load a YAML configuration file; otherwise, use default."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return default

# Define the directory for external configuration files (domain-specific)
CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
INDUSTRY_CONFIGS_FILE = os.path.join(CONFIG_DIR, 'industry_configs.yaml')

# Fallback default industry configurations (if YAML file is not available)
_default_industry_configs = {
    'technology': {
        'name': 'technology',
        'required_sections': [
            'technical skills', 
            'experience',
            'projects',
            'certifications'
        ],
        'skill_categories': list(),  # Leave empty here; YAML should override
        'keywords': ['tech', 'technology', 'it', 'cloud', 'devops', "machine learning", 'full-stack', "artificial intelligence",
                     "ai", "developer tools", 'sdlc', 'api'],
        'experience_patterns': [
            r'\b(developed|implemented|optimized|architected)\b',
            r'\b(microservices|containerization|serverless)\b',
            r'\b(aws|gcp|azure)\b',
            r'\b(CI/CD|automation|pipeline)\b'
        ],
        "section_weights": {"experience": 0.8, "education": 0.2}
    },
    # (Add other industry defaults as needed...)
}

# Load the industry configurations from the external YAML file; fallback if missing.
INDUSTRY_CONFIGS = load_external_config(INDUSTRY_CONFIGS_FILE, _default_industry_configs)

# -------------------------------
# Domain and Technical Constants
# -------------------------------

SKILL_CATEGORIES = {
    # The full SKILL_CATEGORIES dictionary
    'technology': {
        'programming': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'Go', 'Rust', 'Swift', 'Kotlin', 'TypeScript', 'PHP'],
        'web_dev': ['HTML', 'CSS', 'React', 'Angular', 'Vue', 'NodeJS', 'Django', 'Flask', 'Express', 'Laravel', 'Spring', 'ASP.NET'],
        'cloud': ['AWS', 'Amazon Web Services','GCP', 'Azure', 'Kubernetes', 'Docker', 'Terraform', 'Ansible', 'Jenkins', 'Serverless', 'OpenStack', 'AWS Lambda', 'S3'],
        'databases': ['SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'Oracle', 'SQLite', 'Firebase', 'DynamoDB', 'Elasticsearch'],
        'devops': ['CI/CD', 'Git', 'GitHub Actions', 'GitLab CI', 'Jenkins', 'Terraform', 'Ansible', 'Prometheus', 'Grafana', 'ELK Stack', 'ArgoCD'],
        'data': ['Data Analysis', 'Big Data', 'Deep Learning', 'Data Mining', 'Data Visualization', 'Apache Spark', 'Hadoop'],
        'security': ['Cybersecurity', 'Penetration Testing', 'Network Security', 'Cryptography', 'SOC2 Compliance', 'ISO27001 Compliance', 'Zero-Trust Security Framework'],
        "Frontend Frameworks": ["React", "Angular", "Vue.js"],
        "ai_ml": ["Machine Learning", "Artificial Intelligence"],
        "management": ["Project Management", "Agile", "Scrum", "Kanban"], 
        'general': [],  
        'errors': []   
    },
    'engineering': {
        'mechanical_engineering': ['CAD (SolidWorks, AutoCAD, CATIA, Inventor)', 'FEA (Finite Element Analysis - ANSYS, Abaqus)', 'Thermodynamics', 'Fluid Mechanics', 'Heat Transfer', 'Materials Science', 'Manufacturing Processes (CNC Machining, Welding, Casting)', 'Robotics', 'HVAC', 'Mechatronics', 'Kinematics and Dynamics'],
        'electrical_engineering': ['Circuit Design (Analog, Digital)', 'Power Systems (Generation, Transmission, Distribution)', 'Electronics (Microcontrollers, Sensors, Actuators)', 'Signal Processing', 'Embedded Systems', 'Microcontrollers (Arduino, Raspberry Pi, PIC)', 'PCB Design (Altium, Eagle)', 'Control Systems', 'Telecommunications', 'Power Electronics'],
        'civil_engineering': ['Structural Analysis (Steel, Concrete, Timber)', 'Geotechnical Engineering (Soil Mechanics, Foundation Design)', 'Transportation Planning (Highway Design, Traffic Engineering)', 'Water Resources (Hydrology, Hydraulics, Water Treatment)', 'Construction Management', 'CAD (Civil 3D, Revit)', 'Surveying', 'Environmental Engineering (Water Quality, Waste Management)'],
        'chemical_engineering': ['Process Design', 'Reaction Engineering', 'Thermodynamics', 'Mass Transfer', 'Separation Processes', 'Plant Safety', 'Process Control', 'Reaction Kinetics', 'Biochemical Engineering', 'Polymer Engineering', 'Aspen Plus'],
        'aerospace_engineering': ['Aerodynamics', 'Propulsion (Jet Engines, Rocket Engines)', 'Flight Mechanics', 'Spacecraft Design', 'Avionics', 'Orbital Mechanics', 'Aerostructures', 'CFD (Computational Fluid Dynamics)', 'MATLAB', 'STK (Satellite Tool Kit)'],
        'industrial_engineering': ['Process Improvement (Lean Manufacturing, Six Sigma)', 'Operations Research', 'Supply Chain Management', 'Ergonomics', 'Simulation (Arena, AnyLogic)', 'Work Study', 'Facility Layout', 'Inventory Management', 'Project Management'],
        'environmental_engineering': ['Water Treatment', 'Wastewater Treatment', 'Air Pollution Control', 'Solid Waste Management', 'Hazardous Waste Management', 'Environmental Regulations', 'Environmental Impact Assessment', 'Sustainability'],
        'biomedical_engineering': ['Medical Devices (Design, Development)', 'Biomaterials', 'Biomechanics', 'Tissue Engineering', 'Medical Imaging', 'Biosensors', 'Rehabilitation Engineering', 'Clinical Engineering'],
        'systems_engineering': ['System Design', 'System Integration', 'Requirements Management', 'Verification and Validation', 'Risk Management', 'Modeling and Simulation', 'Architecture Frameworks', 'Lifecycle Management'],
        'general': [],  
        'errors': []
    },
    'healthcare': {
        'medical_procedures': ['Surgery', 'Diagnosis', 'Treatment', 'Therapy', 'Rehabilitation', 'Vaccination', 'Intubation'],
        'patient_care': ['Assessment', 'Monitoring', 'Medication Administration', 'Counseling', 'Documentation'],
        'medical_specialties': ['Cardiology', 'Neurology', 'Oncology', 'Pediatrics', 'Orthopedics', 'Psychiatry'],
        'medical_equipment': ['EKG', 'MRI', 'X-ray', 'Ventilator', 'Defibrillator'],
        'regulatory_compliance': ['HIPAA', 'FDA Regulations'],
        'general': [],  
        'errors': []
    },
    'business': {
        'strategy': ['Strategic Planning', 'Business Development', 'Market Analysis', 'Competitive Analysis'],
        'operations': ['Supply Chain Management', 'Logistics', 'Process Optimization', 'Quality Control'],
        'sales_marketing': ['Sales Management', 'Digital Marketing', 'Social Media Marketing', 'Content Marketing', 'SEO', 'SEM'],
        'finance': ['Financial Analysis', 'Budgeting', 'Forecasting', 'Investment Management'],
        'human_resources': ['Talent Acquisition', 'Employee Relations', 'Performance Management', 'Compensation & Benefits'],
        'general': [],  
        'errors': []
    },
    'finance': {
        'financial_analysis': ['Financial Modeling', 'Valuation', 'Risk Management', 'Financial Reporting'],
        'investment_banking': ['Mergers & Acquisitions', 'Initial Public Offerings (IPOs)', 'Debt Financing', 'Equity Financing'],
        'asset_management': ['Portfolio Management', 'Hedge Funds', 'Private Equity'],
        'regulatory_compliance': ['SEC Regulations', 'FINRA Compliance'],
        'financial_instruments': ['Stocks', 'Bonds', 'Derivatives', 'Options', 'Futures'],
        'general': [],  
        'errors': []
    },
    'legal': {
        'litigation': ['Trial Advocacy', 'Appellate Practice', 'Discovery', 'Pleadings'],
        'corporate_law': ['Mergers & Acquisitions', 'Contract Law', 'Intellectual Property Law', 'Securities Law'],
        'criminal_law': ['Criminal Defense', 'Prosecution'],
        'legal_research': ['Case Law Analysis', 'Statutory Interpretation'],
        'regulatory_law': ['Environmental Law', 'Healthcare Law', 'Labor Law'],
        'general': [],  
        'errors': []
    },
    'education': {
        'teaching_methodologies': ['Curriculum Development', 'Lesson Planning', 'Assessment Strategies', 'Differentiated Instruction'],
        'subject_matter_expertise':[], # This would be populated based on the specific educational role
        'classroom_management': ['Behavior Management', 'Creating a Positive Learning Environment'],
        'educational_technology': ['Learning Management Systems (LMS)', 'Interactive Whiteboards'],
        'student_support': ['Counseling', 'Mentoring', 'Tutoring'],
        'general': [],  
        'errors': []
    },
    'creative': {
        'design': ['Graphic Design', 'Web Design', 'UI/UX Design', 'Fashion Design', 'Industrial Design'],
        'visual_arts': ['Painting', 'Sculpture', 'Photography', 'Illustration'],
        'performing_arts': ['Acting', 'Dancing', 'Music Composition', 'Choreography'],
        'writing_editing': ['Copywriting', 'Technical Writing', 'Creative Writing', 'Proofreading', 'Editing'],
        'digital_media': ['Video Production', 'Animation', 'Sound Design'],
        'general': [],  
        'errors': []
    },
    'management': {
        'leadership': ['Team Leadership', 'Strategic Leadership', 'Change Management', 'Decision Making'],
        'project_management': ['Planning', 'Execution', 'Monitoring & Controlling', 'Risk Management'],
        'team_management': ['Delegation', 'Motivation', 'Performance Management', 'Conflict Resolution'],
        'operations_management': ['Process Improvement', 'Efficiency Optimization', 'Resource Allocation'],
        'communication': ['Verbal Communication', 'Written Communication', 'Presentation Skills'],
        'general': [],  
        'errors': []
    },
    'general': {
        'communication': ['Verbal Communication', 'Written Communication', 'Active Listening', 'Interpersonal Skills'],
        'organization': ['Time Management', 'Planning', 'Prioritization', 'Attention to Detail'],
        'problem_solving': ['Critical Thinking', 'Analytical Skills', 'Troubleshooting'],
        'interpersonal_skills': ['Teamwork', 'Collaboration', 'Empathy', 'Negotiation'],
        'technical_aptitude':[], # Can be populated based on context
        'general': [],  
        'errors': []
    },
}

TECH_TERMS = {
    'python', 'java', 'javascript', 'docker', 'kubernetes', 'sql', 'react', 'angular', 'vue', 
    'aws', 'django', 'flask', 'nosql', 'mongodb', 'postgresql', 'mysql', 'git', 'rest', 'api', 
    'nodejs', 'microservices', 'ci/cd', 'cloud', 'devops', 'machine learning', 'nlp', 'data analysis'
}

TECH_SYNONYMS = {
    'node.js': 'Node.js',
    'node js': 'Node.js',
    'nodejs': 'Node.js',
    'reactjs': 'React',
    'react js': 'React',
    'aws cloud': 'AWS',
    'amazon web services': 'AWS',
    'gcp': 'Google Cloud Platform',
    'google cloud': 'Google Cloud Platform',
    'azure': 'Azure',
    'js': 'JavaScript',
    'py': 'Python'
}

STANDARD_SECTIONS = [
    'summary',
    'experience',
    'education',
    'skills',
    'certifications and licenses',
    'projects',
    'awards and honors',
    'publications and presentations',
    'languages'
]

SECTION_SYNONYMS = {
    "career objective": "summary",
    "career summary": "summary",
    "qualifications": "summary",
    "summary of qualifications": "summary",
    "profile": "summary",
    "adult care experience": "experience",
    "childcare experience": "experience",
    "clinical experience": "experience",
    "employment history": "experience",
    "patient care": "experience",
    "professional experience": "experience",
    "work experience": "experience",
    "work history": "experience",
    "academic background": "education",
    "degrees": "education",
    "education background": "education",
    "medical training": "education",
    "residency": "education",
    "certifications": "certifications and licenses",
    "licenses": "certifications and licenses",
    "areas of expertise": "skills",
    "core competencies": "skills",
    "technical skills": "skills",
    "key projects": "projects",
    "projects": "projects",
    "awards": "awards and honors",
    "honors": "awards and honors",
    "publications": "publications and presentations",
    "presentations": "publications and presentations",
    "languages": "languages",
    "education": "education",
    "experience": "experience"
}

HEADER_KEYWORDS = STANDARD_SECTIONS + [
    "career summary", "profile", "adult care experience", "childcare experience",
    "employment history", "work experience", "professional experience", "academic background", 
    "education", "experience", "skills", "training", "certifications", "clinical", "patient care"
]

SECTION_PATTERNS_RAW = [
    (r'^(career\s+summary|professional\s+summary|objective)\s*:?$', 'summary'),
    (r'^(profile)\s*:?$', 'summary'),
    (r'^(adult\s+care\s+experience|childcare\s+experience|clinical\s+experience|volunteer\s+experience)\s*:?$', 'experience'),
    (r'^(employment\s+history|work\s+history|work\s+experience|professional\s+experience)\s*:?$', 'experience'),
    (r'^(education|academic\s+background|medical\s+training|residency)\s*:?$', 'education'),
    (r'^(skills|technical\s+skills|core\s+competencies|areas\s+of\s+expertise)\s*:?$', 'skills'),
    (r'^(certifications\s*&\s*licenses|certifications|licenses)\s*:?$', 'certifications and licenses'),
    (r'^(projects|key\s+projects)\s*:?$', 'projects'),
    (r'^(awards|honors)\s*:?$', 'awards and honors'),
    (r'^(publications|presentations)\s*:?$', 'publications and presentations'),
    (r'^(languages)\s*:?$', 'languages'),
]

SECTION_PATTERNS = [(re.compile(pattern, re.IGNORECASE), name) for pattern, name in SECTION_PATTERNS_RAW]

INDUSTRY_SYNONYMS = {
    'tech': 'technology',
    'scientific': 'science',
    'medical': 'healthcare',
    'business': 'business',
    'legal': 'legal',
    'educational': 'education',
    'managerial': 'management',
    'research': 'research',
    'creative': 'creative',
    'ai': 'artificial intelligence',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'nlp': 'natural language processing',
    'iot': 'internet of things',
    'vr': 'virtual reality',
    'ar': 'augmented reality',
    'xr': 'extended reality',
    'db': 'database',
    'dba': 'database administrator',
    'sre': 'site reliability engineering',
    'sdlc': 'software development lifecycle',
    "machine learning research": "technology",
    "ml research": "technology",
    'data sci': 'data science',
    'data eng': 'data engineering',
    'bi': 'business intelligence',
    'etl': 'data pipelines',
    'data viz': 'data visualization',
    'data ops': 'data operations',
    'cloud ops': 'cloud operations',
    'iaas': 'infrastructure as a service',
    'paas': 'platform as a service',
    'saas': 'software as a service',
    'fintech': 'finance',
    'healthtech': 'healthcare',
    'edtech': 'education',
    'legaltech': 'legal',
    'agritech': 'agriculture',
    'govtech': 'government',
    'insurtech': 'insurance',
    'hrtech': 'human resources',
    'proptech': 'real estate',
    'retailtech': 'retail',
    'crm': 'customer relationship management',
    'erp': 'enterprise resource planning',
    'kpi': 'performance metrics',
    'roi': 'return on investment',
    'b2b': 'business-to-business',
    'b2c': 'business-to-consumer',
    'crypto': 'cryptocurrency',
    'defi': 'decentralized finance',
    'finance': 'finance',
    'emr': 'electronic medical records',
    'ehr': 'electronic health records',
    'telemed': 'telemedicine',
    'pharma': 'pharmaceuticals',
    'medtech': 'medical technology',
    'crc': 'clinical research coordinator',
    "healthcare": "healthcare",
    'healthcare technology': 'healthcare',
    'biotech': 'biotechnology',
    'chembio': 'chemical biology',
    'geneng': 'genetic engineering',
    'nano': 'nanotechnology',
    'r&d': 'research and development',
    'gmp': 'good manufacturing practice',
    'ui/ux': 'user interface/experience',
    'uxd': 'user experience design',
    'cad': 'computer-aided design',
    'cgi': 'computer-generated imagery',
    'mograph': 'motion graphics',
    'vidprod': 'video production',
    'lms': 'learning management system',
    'mooc': 'massive open online course',
    'stem': 'science/technology/engineering/math',
    'elearn': 'e-learning',
    'k12': 'primary/secondary education',
    'infosec': 'information security',
    'soc': 'security operations center',
    'iam': 'identity and access management',
    'pentest': 'penetration testing',
    'devsecops': 'security-integrated devops',
    'mobiledev': 'mobile development',
    'webdev': 'web development',
    'lowcode': 'low-code development',
    'nocode': 'no-code development',
    'cad/cam': 'computer-aided manufacturing',
    'plm': 'product lifecycle management',
    'scada': 'industrial control systems',
    'mes': 'manufacturing execution systems',
    'esg': 'environmental/social/governance',
    'ghg': 'greenhouse gas management',
    'circecon': 'circular economy',
    'renewables': 'renewable energy systems',
    'ecommerce': 'retail',
    'e-commerce': 'retail',
    'online retail': 'retail',
    'online store': 'retail',
    'digital commerce': 'retail',
    'eng': 'engineering',
    'engineer': 'engineering',
    'mechanical eng': 'mechanical engineering',
    'electrical eng': 'electrical engineering',
    'civil eng': 'civil engineering',
    'chemical eng': 'chemical engineering',
    'software eng': 'software engineering',
    'aero eng': 'aerospace engineering',
    'industrial eng': 'industrial engineering',
    'environmental eng': 'environmental engineering',
    'biomedical eng': 'biomedical engineering',
    'systems eng': 'systems engineering',
    'mech eng': 'mechanical engineering',
    'ee': 'electrical engineering',
    'ce': 'civil engineering',
    'che': 'chemical engineering',
    'swe': 'software engineering',
    'aero': 'aerospace engineering',
    'ie': 'industrial engineering',
    'env eng': 'environmental engineering',
    'bio eng': 'biomedical engineering',
    'sys eng': 'systems engineering',
    'structural eng': 'civil engineering',
    'geotechnical eng': 'civil engineering',
    'transportation eng': 'civil engineering',
    'water resources eng': 'civil engineering',
    'construction mgmt': 'civil engineering',
    'process eng': 'chemical engineering',
    'reaction eng': 'chemical engineering',
    'materials sci': 'mechanical engineering',
    'fluid dynamics': 'mechanical engineering',
    'thermodynamics': 'mechanical engineering',
    'power systems eng': 'electrical engineering',
    'electronics eng': 'electrical engineering',
    'embedded systems eng': 'electrical engineering',
    'control systems eng': 'electrical engineering',
    'avionics eng': 'aerospace engineering',
    'propulsion eng': 'aerospace engineering',
    'flight mechanics eng': 'aerospace engineering',
    'operations research': 'industrial engineering',
    'supply chain mgmt': 'industrial engineering',
    'ergonomics': 'industrial engineering',
    'water treatment eng': 'environmental engineering',
    'air quality eng': 'environmental engineering',
    'waste management eng': 'environmental engineering',
    'medical device eng': 'biomedical engineering',
    'bioengineering': 'biomedical engineering',
    'biomechanics': 'biomedical engineering',
    'tissue engineering': 'biomedical engineering',
    'robotics eng': 'mechanical engineering',
    'automation eng': 'electrical engineering',
    'cad/cam eng': 'engineering',
    'hvac eng': 'mechanical engineering',
    'manufacturing eng': 'mechanical engineering',
    'design eng': 'engineering',
    'test eng': 'engineering',
    'quality eng': 'engineering',
    'reliability eng': 'engineering',
    'maintenance eng': 'engineering',
    'project eng': 'engineering',
    'product eng': 'engineering'
}

KNOWN_SKILLS = [
    "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust", "Swift", "Kotlin", 
    "TypeScript", "PHP", "HTML", "CSS", "React", "Angular", "Vue", "NodeJS", "Django", 
    "Flask", "Express", "Laravel", "Spring", "ASP.NET", "AWS", "GCP", "Azure", "Kubernetes", 
    "Docker", "Terraform", "Ansible", "Jenkins", "Serverless", "OpenStack", "AWS Lambda", 
    "S3", "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "Oracle", "SQLite", 
    "Firebase", "DynamoDB", "Elasticsearch", "CI/CD", "Git", "GitHub Actions", "GitLab CI", 
    "Prometheus", "Grafana", "ELK Stack", "ArgoCD", "Machine Learning", "Natural Language Processing", 
    "Data Analysis", "Big Data", "Deep Learning", "Data Mining", "Data Visualization", 
    "Apache Spark", "Hadoop", "Cybersecurity", "Penetration Testing", "Network Security", 
    "Cryptography", "Zero-Trust Security Framework", "Node.js", "C++", ".NET", "TensorFlow", 
    "PyTorch", "Apache Kafka", "Azure DevOps", "RESTful APIs", "Infrastructure as Code",
    # Science
    "PCR", "Gel Electrophoresis", "Cell Culture", "Chromatography", "DNA Sequencing", 
    "Flow Cytometry", "CRISPR", "UV-Vis Spectrometer", "HPLC", "Nano Drop", "Gas Chromatograph", 
    "Centrifuge", "Microscopy", "Mass Spectrometer", "Biochemistry", "Molecular Biology", 
    "Organic Chemistry", "Physics", "Genetics", "Neuroscience", "Quantum Computing",
    # Healthcare
    "Patient Care", "Clinical Assessment", "Medical Diagnosis", "Surgical Assistance", 
    "Phlebotomy", "Vaccine Administration", "Anatomy", "Pharmacology", "Medical Ethics", 
    "Epidemiology", "HIPAA Compliance", "Patient Safety", "EHR", "EPIC", "Cerner", 
    "Telehealth", "PACS Systems", "Health Informatics", "CPR", "BLS", "ACLS", "RN", 
    "EMT", "Medical Coding Certification",
    # Business
    "Financial Analysis", "Tax Preparation", "Auditing", "GAAP Compliance", "Portfolio Management", 
    "Cryptocurrency", "Blockchain", "Venture Capital", "Retail Banking", "Fintech", 
    "Fraud Detection", "Commercial Banking", "SEO", "SEM", "Social Media Marketing", 
    "Content Marketing", "Google Analytics", "Brand Management", "Market Research", 
    "Event Planning", "Product Launches", "Negotiation", "Cold Calling", "Sales Strategy", 
    "Consultative Selling", "B2B Sales", "SaaS Sales", "Enterprise Sales", "Retail Sales",
    # Legal
    "Legal Research", "Litigation", "Contract Law", "Arbitration", "Mediation", 
    "Corporate Law", "Intellectual Property", "Real Estate Law", "Cyber Law", 
    "Westlaw", "Clio", "E-Discovery Tools", "Document Automation",
    # Education
    "Curriculum Development", "Online Teaching", "Student Engagement", "Blended Learning", 
    "STEM Education", "E-Learning", "Special Education", "Adult Education", 
    "Educational Leadership", "Accreditation Processes", "Grant Writing", 
    "Community Outreach",
    # Management
    "Agile", "Scrum", "Risk Management", "Resource Allocation", "Waterfall Methodology", 
    "Process Improvement", "KPI Tracking", "Market Analysis", "Vendor Management", 
    "Recruitment", "Employee Relations", "Talent Acquisition", "Workforce Planning", "Strategic Planning",
    # Research
    "Quantitative Research", "Qualitative Research", "Survey Design", "Meta-Analysis", 
    "R", "Python Pandas", "Tableau", "MATLAB", "SPSS", "Grant Writing", "Peer Review", 
    "Academic Writing", "Research Ethics",
    # Creative
    "Photoshop", "Figma", "Blender", "Premiere Pro", "AutoCAD", "UI Design", 
    "Graphic Design", "Prototyping", "User Research", "Brand Design", "Motion Design", 
    "Game Design", "3D Modeling", "Video Editing", "Animation", "Photography", 
    "Copywriting", "Storytelling",
    # General/Soft Skills
    "Leadership", "Communication", "Teamwork", "Problem-Solving", "Time Management", 
    "Adaptability", "Critical Thinking", "Empathy", "Conflict Resolution", 
    "English", "Spanish", "French", "Mandarin", "German"
]

SEED_VERBS = {
    'technology': [
        'develop', 'architect', 'implement', 'code', 'debug', 'test', 'deploy',
        'integrate', 'optimize', 'troubleshoot', 'upgrade', 'automate', 'prototype',
        'configure', 'migrate', 'secure', 'virtualize', 'containerize'
    ],
    'engineering': [
        'design', 'analyze', 'build', 'test', 'maintain', 'repair', 'optimize', 'troubleshoot',
        'install', 'inspect', 'calculate', 'simulate', 'model', 'develop', 'implement', 'engineer',
        'construct', 'fabricate', 'innovate', 'conceptualize', 'validate', 'verify', 'specify',
        'detail', 'architect', 'commission', 'weld', 'machine', 'mold', 'cast', 'calibrate',
        'overhaul', 'upgrade', 'plan', 'coordinate', 'document', 'standardize', 'improve',
        'automate', 'program', 'configure', 'integrate', 'deploy', 'sustain'
    ],
    'healthcare': [
        'diagnose', 'treat', 'monitor', 'prescribe', 'administer', 'assess',
        'vaccinate', 'rehabilitate', 'counsel', 'sanitize', 'sterilize', 'intubate',
        'triangulate', 'educate', 'document', 'collaborate', 'evaluate', 'immunize'
    ],
    'business': [
        'negotiate', 'optimize', 'scale', 'strategize', 'analyze', 'innovate',
        'facilitate', 'invest', 'acquire', 'outsource', 'benchmark', 'diversify',
        'restructure', 'monetize', 'pivot', 'network', 'forecast', 'streamline'
    ],
    'finance': [
        'analyze', 'forecast', 'budget', 'allocate', 'audit', 'reconcile',
        'invest', 'hedge', 'underwrite', 'evaluate', 'report', 'mitigate',
        'accrue', 'depreciate', 'model', 'comply', 'leverage', 'rebalance'
    ],
    'legal': [
        'draft', 'litigate', 'advise', 'review', 'mediate', 'file',
        'argue', 'notarize', 'research', 'counsel', 'settle', 'appeal',
        'negotiate', 'investigate', 'copyright', 'trademark', 'arbitrate', 'legislate'
    ],
    'education': [
        'teach', 'mentor', 'develop', 'instruct', 'assess', 'plan',
        'tutor', 'facilitate', 'inspire', 'evaluate', 'guide', 'adapt',
        'differentiate', 'moderate', 'demonstrate', 'encourage', 'scaffold', 'accredit'
    ],
    'creative': [
        'design', 'illustrate', 'produce', 'create', 'conceptualize', 'edit',
        'direct', 'photograph', 'write', 'compose', 'prototype', 'animate',
        'storyboard', 'curate', 'brand', 'choreograph', 'sculpt', 'typograph'
    ],
    'management': [
        'lead', 'streamline', 'coordinate', 'supervise', 'delegate', 'oversee',
        'evaluate', 'motivate', 'align', 'empower', 'mentor', 'orchestrate',
        'prioritize', 'restructure', 'allocate', 'resolve', 'standardize', 'synergize'
    ],
    'general': [
        'manage', 'improve', 'organize', 'execute', 'plan', 'coordinate',
        'oversee', 'enhance', 'solve', 'communicate', 'prioritize', 'facilitate',
        'monitor', 'document', 'review', 'update', 'maintain', 'track'
    ]
}

WEAK_VERBS = {
    "helped", "supported", "assisted", "made", "did", "worked on",
    "handled", "used", "followed", "tried", "participated", 
    "contributed", "provided", "maintained", "involved", 
    "aided", "liaised", "reacted", "dealt with", "helped with",
    "gave", "looked after", "completed", "helped out", 
    "was responsible for", "facilitated", "coordinated",
    "engaged in", "processed", "did work on", "assisted with",
    "debugged", "researched", "answered", "inputted", "shadowed",
}

GENERAL_FEEDBACK_RULES = {
    'word_count': {
        'high': 800,
        'low': 300,
        'high_message': "Resume is too long. Trim to 1-2 pages.",
        'low_message': "Resume is too short. Add more details.",
    },
    "skill_balance": "Balance technical skills with more soft skills (recommended 60% soft skills).",
    'experience_threshold': 2,  # Minimum years to avoid "entry-level" warnings
    'metric_guidance': "At least 70% of achievements should include quantifiable metrics",
    'verb_check': "Ensure 90% of bullet points start with action verbs",
    "keyword_density": 0.07,
    "passive_voice": {
        "threshold": 1,
        "message": "Reduce passive voice ({count} found). Use active voice for impact."
    },
    "ml_suggestions": {
        "max_skills": 3,
        "max_achievements": 2
    }
}
