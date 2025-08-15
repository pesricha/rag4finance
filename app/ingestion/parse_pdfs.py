import os
import re
from PyPDF2 import PdfReader
from typing import List, Dict

MONTH_PATTERN = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"

def parse_pdf(file_path: str) -> str:
    """Extract full text from a PDF using PyPDF2."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text

def chunk_by_month(text: str, source: str) -> List[Dict]:
    """Split text into chunks by month-year headings."""
    matches = list(re.finditer(MONTH_PATTERN, text, re.IGNORECASE))
    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        month_str = match.group(0)
        chunk_text = text[start:end].strip()

        chunks.append({
            "month": month_str,
            "text": chunk_text,
            "source": source
        })

    return chunks

def parse_all_pdfs_by_month(folder_path: str) -> List[Dict]:
    """Read all PDFs in a folder and return monthly chunks."""
    all_chunks = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            print(f"Processing filepath: {file_path}")

            text = parse_pdf(file_path)
            # print(f"text: {text}...")  # Print first 100 characters for debugging
            # import sys
            # sys.exit("Exiting after parsing one file for debugging purposes.")  
            monthly_chunks = chunk_by_month(text, source=file)
            # print(f"monthly_chunks: {len(monthly_chunks)}")
            # import sys
            # sys.exit("Exiting after processing one file for debugging purposes.")
            all_chunks.extend(monthly_chunks)
            print(f"✅ Processed {file} → {len(monthly_chunks)} month chunks")

    print(f"Total files processed: {len(all_chunks)}")
    return all_chunks

# if __name__ == "__main__":
#     folder = "/home/biomedialab/Desktop/Sandeep/Placements/Projects/rag4finance/data"  # change this
#     monthly_data = parse_all_pdfs_by_month(folder)
#     print(f"\nTotal chunks: {len(monthly_data)}")
