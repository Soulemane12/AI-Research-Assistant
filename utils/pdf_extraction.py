import fitz  # PyMuPDF
import logging

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        logging.info(f"Text extracted successfully from {uploaded_file.name}.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {uploaded_file.name}: {e}")
        return ""
