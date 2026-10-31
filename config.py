import os
from dotenv import load_dotenv
load_dotenv()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
