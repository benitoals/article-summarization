import os
from PyPDF2 import PdfReader

def pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            # Iterate through all the pages in the PDF
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def convert_pdfs_in_folder(folder_path):
    """Convert all PDF files in the given folder to TXT files."""
    # Iterate over every file in the directory
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            text = pdf_to_text(pdf_path)
            if text:
                # Create a new filename by replacing .pdf with .txt
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(folder_path, txt_filename)
                try:
                    with open(txt_path, "w", encoding="utf-8") as txt_file:
                        txt_file.write(text)
                    print(f"Converted {filename} to {txt_filename}")
                except Exception as e:
                    print(f"Error writing {txt_filename}: {e}")
            else:
                print(f"No text extracted from {filename}.")

if __name__ == '__main__':
    folder_path = input("Enter the folder path containing the PDF files: ").strip()
    if os.path.isdir(folder_path):
        convert_pdfs_in_folder(folder_path)
    else:
        print("Invalid folder path. Please check and try again.")