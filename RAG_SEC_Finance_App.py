# -*- coding: utf-8 -*-
"""
SEC Financial RAG Application
A Flask-based application for analyzing SEC 10-K filings using RAG (Retrieval Augmented Generation)
"""

import os
import json
import logging
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil

from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
USER_EMAIL = os.getenv('USER_EMAIL', 'your-email@example.com')
PROJECT_NAME = os.getenv('PROJECT_NAME', 'SEC RAG Application')

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('logs/sec_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Directories
WORK_DIR = Path("work")
UPLOADS_DIR = Path("uploads")
INDEX_DIR = WORK_DIR / "index"
LOGS_DIR = Path("logs")

# Create directories
for directory in [WORK_DIR, UPLOADS_DIR, INDEX_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Configure LLM and embeddings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# Company ticker to CIK mapping (expand as needed)
COMPANY_CIK_MAP = {
    'AAPL': '0000320193',
    'MSFT': '0000789019',
    'GOOGL': '0001652044',
    'AMZN': '0001018724',
    'TSLA': '0001318605',
    'META': '0001326801',
    'NVDA': '0001045810',
    'JPM': '0000019617',
    'JNJ': '0000200406',
    'PG': '0000080424',
    'BAC': '0000070858',
    'XOM': '0000034088',
    'WMT': '0000104169',
    'CVX': '0000093410',
    'UNH': '0000731766'
}

class SECDataExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'{PROJECT_NAME} {USER_EMAIL}',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
    
    def get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK from ticker symbol"""
        ticker = ticker.upper().strip()
        return COMPANY_CIK_MAP.get(ticker)
    
    def fetch_company_submissions(self, cik: str) -> Dict[str, Any]:
        """Fetch company submissions from SEC API"""
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        logger.info(f"Fetching submissions index from: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching submissions: {e}")
            raise
    
    def find_latest_10k(self, submissions: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Find the latest 10-K filing"""
        filings = submissions.get('filings', {}).get('recent', {})
        forms = filings.get('form', [])
        accession_numbers = filings.get('accessionNumber', [])
        filing_dates = filings.get('filingDate', [])
        primary_documents = filings.get('primaryDocument', [])
        
        for i, form in enumerate(forms):
            if form == '10-K':
                return {
                    'accession': accession_numbers[i],
                    'date': filing_dates[i],
                    'primary_doc': primary_documents[i]
                }
        logger.warning("No 10-K filing found")
        return None
    
    def download_10k_document(self, cik: str, accession: str, primary_doc: str) -> str:
        """Download 10-K document and return as text"""
        # Clean up accession number for URL
        accession_clean = accession.replace('-', '')
        
        url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_doc}"
        logger.info(f"Downloading primary document from: {url}")
        
        # Update headers for SEC website
        headers = {
            'User-Agent':  f'{PROJECT_NAME} {USER_EMAIL}',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # Parse HTML and extract meaningful content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Enhanced text extraction focusing on financial sections
            text_content = self.extract_financial_sections(soup)
            
            # Save to file
            filename = f"{cik}_10K_{accession}_{primary_doc.replace('.htm', '.txt')}"
            filepath = UPLOADS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            logger.info(f"Successfully downloaded and converted 10-K HTML to text: {filepath}")
            return str(filepath)
            
        except requests.RequestException as e:
            logger.error(f"Error downloading document: {e}")
            raise
    
    def extract_financial_sections(self, soup: BeautifulSoup) -> str:
        """Enhanced extraction of financial sections from 10-K HTML"""
        sections = []
        
        # Extract text from all elements
        all_text = soup.get_text()
        
        # Clean up the text
        lines = []
        for line in all_text.split('\n'):
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                # Clean up excessive whitespace
                line = re.sub(r'\s+', ' ', line)
                lines.append(line)
        
        text_content = '\n'.join(lines)
        
        # Try to extract specific financial sections
        financial_content = []
        
        # Look for financial tables and key financial information
        tables = soup.find_all('table')
        for table in tables:
            table_text = table.get_text()
            # Check if this table contains financial data
            if any(keyword in table_text.lower() for keyword in 
                   ['revenue', 'income', 'expense', 'assets', 'liabilities', 'equity', 
                    'cash flow', 'earnings', 'profit', 'loss', 'million', 'billion']):
                financial_content.append(table_text)
        
        # If we found financial tables, prioritize them
        if financial_content:
            sections.extend(financial_content)
        
        # Add the cleaned full text
        sections.append(text_content)
        
        return '\n\n'.join(sections)

class DocumentProcessor:
    def __init__(self):
        self.llm = Settings.llm
    
    def extract_text_from_file(self, filepath: str) -> str:
        """Extract text content from file"""
        logger.info(f"Extracting content from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Successfully extracted text from {Path(filepath).name}")
            return content
            
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {e}")
            raise
    
    def save_as_markdown(self, content: str, filepath: str) -> str:
        """Save content as markdown file"""
        md_dir = WORK_DIR / Path(filepath).stem
        md_dir.mkdir(exist_ok=True)
        
        md_filepath = md_dir / f"{Path(filepath).stem}.md"
        
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Financial Document Analysis\n\n")
            f.write(f"Source: {Path(filepath).name}\n\n")
            f.write(content)
        
        logger.info(f"Saved text to Markdown: {md_filepath}")
        return str(md_filepath)
    
    def summarize_document(self, md_filepath: str) -> str:
        """Generate summary of the document focusing on financial data"""
        logger.info(f"Summarizing Markdown file: {md_filepath}")
        
        try:
            with open(md_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Truncate content if too large (keep first 8000 chars for summary)
            if len(content) > 8000:
                content = content[:8000] + "...[truncated]"
            
            prompt = f"""
            Analyze this 10-K filing excerpt and provide a comprehensive summary focusing on:
            
            1. **Revenue**: Specific revenue figures, year-over-year changes, and revenue segments
            2. **Net Income**: Profit/loss figures and trends
            3. **Major Expenses or Cost Drivers**: Key expense categories and their amounts
            4. **Significant Trends or Outlook Statements**: Forward-looking statements and business outlook
            5. **Key Financial Metrics**: Any important ratios, margins, or financial indicators mentioned
            
            Content to analyze:
            {content}
            
            Provide a detailed summary with actual numbers and financial data where available:
            """
            
            response = self.llm.complete(prompt)
            summary = str(response)
            
            # Save summary
            summary_filepath = Path(md_filepath).parent / f"{Path(md_filepath).stem}_summary.md"
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Document Summary\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(summary)
            
            logger.info(f"Successfully generated summary for {Path(md_filepath).name}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed."

class RAGSystem:
    def __init__(self):
        self.index = None
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing index or create new empty index"""
        try:
            if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
                logger.info(f"Loading existing index from {INDEX_DIR}")
                storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
                self.index = load_index_from_storage(storage_context)
                logger.info("Existing index loaded successfully.")
            else:
                logger.info("No existing index found. Building new empty index.")
                self.build_new_index([])
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Building new empty index.")
            self.build_new_index([])
    
    def build_new_index(self, documents: List[Document]):
        """Build new vector index"""
        logger.info("Building new vector index...")
        
        if not documents:
            # Create a dummy document to initialize the index
            documents = [Document(text="Initial empty index document.", doc_id="initial")]
        
        self.index = VectorStoreIndex.from_documents(documents)
        self.persist_index()
        logger.info(f"Vector index built and persisted to {INDEX_DIR}")
    
    def persist_index(self):
        """Persist index to disk"""
        self.index.storage_context.persist(persist_dir=str(INDEX_DIR))
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing index"""
        if self.index is None:
            self.build_new_index(documents)
        else:
            # Get all documents including existing ones
            storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
            existing_docs = []
            
            try:
                # Load existing documents
                docstore = storage_context.docstore
                existing_docs = [docstore.get_document(doc_id) for doc_id in docstore.get_all_document_ids()]
            except:
                pass
            
            # Combine existing and new documents
            all_documents = existing_docs + documents
            self.build_new_index(all_documents)
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if self.index is None:
            return "No documents have been ingested yet. Please ingest some documents first."
        
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize"
            )
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return f"An error occurred while processing your query: {str(e)}"

# Initialize components
sec_extractor = SECDataExtractor()
doc_processor = DocumentProcessor()
rag_system = RAGSystem()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Docker"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/ingest', methods=['POST'])
def ingest_document():
    """Ingest SEC documents"""
    try:
        data = request.json
        ingest_type = data.get('type')
        value = data.get('value', '').strip()
        
        logger.info(f"Received ingest request: type='{ingest_type}', value='{value[:20]}...'")
        
        if ingest_type == 'cik':
            # Check if it's a ticker symbol first
            cik = sec_extractor.get_cik_from_ticker(value)
            if cik:
                logger.info(f"Recognized ticker '{value}', using CIK: {cik}")
            else:
                # Assume it's already a CIK
                cik = value.zfill(10)  # Pad with zeros to 10 digits
            
            # Download and process 10-K
            logger.info(f"Attempting to download 10-K for CIK: {cik}")
            
            submissions = sec_extractor.fetch_company_submissions(cik)
            latest_10k = sec_extractor.find_latest_10k(submissions)
            
            if not latest_10k:
                return jsonify({'error': 'No 10-K filing found for this company'}), 404
            
            logger.info(f"Found 10-K: Accession '{latest_10k['accession']}', Date '{latest_10k['date']}'")
            
            # Download the document
            filepath = sec_extractor.download_10k_document(
                cik, 
                latest_10k['accession'], 
                latest_10k['primary_doc']
            )
            
        else:
            return jsonify({'error': 'Invalid ingest type'}), 400
        
        # Process the document
        content = doc_processor.extract_text_from_file(filepath)
        md_filepath = doc_processor.save_as_markdown(content, filepath)
        summary = doc_processor.summarize_document(md_filepath)
        
        # Create documents for RAG
        documents = [
            Document(
                text=content,
                doc_id=f"{Path(filepath).stem}_full",
                metadata={
                    'filename': Path(filepath).name,
                    'type': 'full_document',
                    'cik': cik if ingest_type == 'cik' else None,
                    'filing_date': latest_10k['date'] if ingest_type == 'cik' else None
                }
            ),
            Document(
                text=summary,
                doc_id=f"{Path(filepath).stem}_summary",
                metadata={
                    'filename': Path(filepath).name,
                    'type': 'summary',
                    'cik': cik if ingest_type == 'cik' else None,
                    'filing_date': latest_10k['date'] if ingest_type == 'cik' else None
                }
            )
        ]
        
        # Add to RAG system
        rag_system.add_documents(documents)
        
        return jsonify({
            'message': '✅ 10-K successfully ingested!',
            'summary': summary,
            'filename': Path(filepath).name,
            'filing_date': latest_10k['date'] if ingest_type == 'cik' else None
        })
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_documents():
    """Query the RAG system"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        logger.info(f"Received query: {question[:100]}...")
        
        # Query the RAG system
        answer = rag_system.query(question)
        
        logger.info("Query processed successfully")
        
        return jsonify({
            'answer': answer,
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('APP_PORT', 8000))
    host = os.getenv('APP_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting SEC RAG Application on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

