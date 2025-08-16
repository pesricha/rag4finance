import os
import re
from PyPDF2 import PdfReader
from typing import List, Dict
import tiktoken

MONTH_PATTERN = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"

class PDFChunker:
    def __init__(self, chunk_size: int = 350, overlap: int = 75, encoding_name: str = "cl100k_base"):
        """
        Initialize the PDF chunker with token-based parameters.
        
        Args:
            chunk_size: Target chunk size in tokens (300-400 recommended)
            overlap: Overlap between chunks in tokens (50-100 recommended)
            encoding_name: Tiktoken encoding to use for token counting
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def parse_pdf(self, file_path: str) -> str:
        """Extract full text from a PDF using PyPDF2."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
        return text
    
    def split_text_by_tokens(self, text: str, preserve_paragraphs: bool = True) -> List[str]:
        """
        Split text into chunks based on token count with overlap.
        
        Args:
            text: Input text to split
            preserve_paragraphs: Try to split at paragraph boundaries when possible
        
        Returns:
            List of text chunks
        """
        # Split text into sentences for better chunking boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self.get_overlap_text(current_chunk, self.overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            
            i += 1
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens from text for overlap."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_token_ids = tokens[-overlap_tokens:]
        return self.encoding.decode(overlap_token_ids)
    
    def chunk_monthly_data(self, month_text: str, month_str: str) -> List[Dict]:
        """
        Break a single month's data into chunks of ~350 tokens with 75 token overlap.
        
        Args:
            month_text: Text content for the month
            month_str: Month identifier (e.g., "January 2024")
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Remove the month header from the text to avoid duplication
        clean_text = re.sub(f"^{re.escape(month_str)}", "", month_text).strip()
        
        if not clean_text:
            return chunks
        
        # Split into words for more granular control
        words = clean_text.split()
        
        if not words:
            return chunks
        
        chunk_index = 0
        start_idx = 0
        
        while start_idx < len(words):
            # Build chunk starting from start_idx
            current_chunk_words = []
            current_tokens = 0
            
            # Add words until we reach target chunk size
            for i in range(start_idx, len(words)):
                word = words[i]
                # Test adding this word
                test_chunk = " ".join(current_chunk_words + [word])
                test_tokens = self.count_tokens(test_chunk)
                
                if test_tokens <= self.chunk_size:
                    current_chunk_words.append(word)
                    current_tokens = test_tokens
                else:
                    # We've reached the size limit
                    break
            
            # Create the chunk
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append({
                    "month": month_str,
                    "chunk_id": f"{month_str}_chunk_{chunk_index + 1}",
                    "text": chunk_text,
                    "token_count": current_tokens,
                })
                
                chunk_index += 1
                
                # Calculate overlap for next chunk
                if start_idx + len(current_chunk_words) < len(words):  # Not the last chunk
                    # Find the overlap point (75 tokens from the end of current chunk)
                    overlap_words = []
                    overlap_tokens = 0
                    
                    # Work backwards from end of current chunk to find overlap
                    for j in range(len(current_chunk_words) - 1, -1, -1):
                        test_overlap = " ".join(current_chunk_words[j:])
                        test_tokens = self.count_tokens(test_overlap)
                        
                        if test_tokens <= self.overlap:
                            overlap_words = current_chunk_words[j:]
                            overlap_tokens = test_tokens
                        else:
                            break
                    
                    # Next chunk starts with overlap
                    if overlap_words:
                        # Find where overlap ends in the word list
                        overlap_length = len(overlap_words)
                        start_idx = start_idx + len(current_chunk_words) - overlap_length
                    else:
                        start_idx = start_idx + len(current_chunk_words)
                else:
                    # Last chunk, we're done
                    break
            else:
                # If we couldn't add any words, move forward to avoid infinite loop
                start_idx += 1
        
        return chunks

    def processing_by_month(self, text: str) -> List[Dict]:
        """Split text by month-year headings, then chunk each month's data."""
        matches = list(re.finditer(MONTH_PATTERN, text, re.IGNORECASE))
        all_chunks = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            month_str = match.group(0)
            month_text = text[start:end].strip()
            
            # Chunk this month's data
            month_chunks = self.chunk_monthly_data(month_text, month_str)
            all_chunks.extend(month_chunks)
            
            if month_chunks:
                total_tokens = sum(chunk["token_count"] for chunk in month_chunks)
                print(f"  📅 {month_str}: {len(month_chunks)} chunks, {total_tokens} tokens")

        return all_chunks
    
    def process_single_pdf(self, file_path: str) -> List[Dict]:
        """Process a single PDF file and return chunked data by month."""
        print(f"Processing: {os.path.basename(file_path)}")
        
        text = self.parse_pdf(file_path)
        monthly_chunks = self.processing_by_month(text)
        
        # Add file metadata to each chunk
        filename = os.path.basename(file_path)
        
        print(f"  📄 Total chunks created: {len(monthly_chunks)}")
        return monthly_chunks
    
    def parse_all_pdfs_by_month(self, folder_path: str) -> List[Dict]:
        """Read all PDFs in a folder and return monthly chunks with token-based splitting."""
        all_chunks = []
        
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            print("No PDF files found in the specified folder.")
            return all_chunks
        
        for file in pdf_files:
            file_path = os.path.join(folder_path, file)
            
            try:
                monthly_chunks = self.process_single_pdf(file_path)
                all_chunks.extend(monthly_chunks)
                
                total_tokens = sum(chunk["token_count"] for chunk in monthly_chunks)
                print(f"✅ {file} → {len(monthly_chunks)} chunks, {total_tokens} total tokens")
                
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        token_counts = [chunk["token_count"] for chunk in chunks]
        months = list(set(chunk["month"] for chunk in chunks))
        
        return {
            "total_chunks": len(chunks),
            "unique_months": len(months),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "months_covered": sorted(months)
        }

# Example usage and utility functions
def main():
    """Example usage of the PDFChunker class."""
    # Initialize chunker with your desired parameters
    chunker = PDFChunker(
        chunk_size=350,    # Target 350 tokens per chunk
        overlap=75,        # 75 token overlap between chunks
    )
    
    # Process all PDFs in a folder
    folder = "./data"  # Update this path
    monthly_data = chunker.parse_all_pdfs_by_month(folder)
    
    # Get statistics
    stats = chunker.get_chunk_statistics(monthly_data)

    print(f"type(monthly_data): {type(monthly_data)}")
    print(f"first chunk: {monthly_data[-1]}")
    
    print(f"\n📊 Processing Statistics:")
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Average tokens per chunk: {stats.get('avg_tokens_per_chunk', 0):.1f}")
    print(f"Token range: {stats.get('min_tokens', 0)}-{stats.get('max_tokens', 0)}")
    print(f"Total tokens: {stats.get('total_tokens', 0):,}")
    print(f"Months covered: {len(stats.get('months_covered', []))}")
    
    # Group chunks by month for analysis
    chunks_by_month = {}
    for chunk in monthly_data:
        month = chunk['month']
        if month not in chunks_by_month:
            chunks_by_month[month] = []
        chunks_by_month[month].append(chunk)
    
    print(f"\n📅 Chunks per month:")
    for month, chunks in sorted(chunks_by_month.items()):
        total_tokens = sum(c['token_count'] for c in chunks)
        print(f"  {month}: {len(chunks)} chunks, {total_tokens} tokens")
    
    # Example: Print first chunk from each month
    print(f"\n📝 Sample chunks (first chunk from each month):")
    seen_months = set()
    for chunk in monthly_data:
        if chunk['month'] not in seen_months:
            seen_months.add(chunk['month'])
            print(f"Tokens: {chunk['token_count']}")
            print(f"Text preview: {chunk['text'][:150]}...")
            if len(seen_months) >= 3:  # Show max 3 examples
                break

if __name__ == "__main__":
    main()
