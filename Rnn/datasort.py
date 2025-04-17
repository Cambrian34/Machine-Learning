import docx
import fitz  # PyMuPDF
import os

# === Function to extract text from PDF ===
def extract_text_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === Function to extract text from DOCX ===
def extract_text_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# === Function to read simple TXT files ===
def extract_text_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# === General function to read any file type ===
def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_pdf(file_path)
    elif ext == "docx":
        return extract_text_docx(file_path)
    elif ext == "txt":
        return extract_text_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")