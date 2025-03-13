import os
import time
import logging
import json
import uuid
import io
from typing import Dict, Any, List, Optional, BinaryIO
from PyPDF2 import PdfReader
import redis
from loguru import logger

class PDFController:
    """Controller for PDF document processing"""
    
    def __init__(self):
        """Initialize the PDF controller and Redis connection"""
        # Set up Redis connection
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        
        try:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True  # Auto-decode Redis responses to strings
            )
            # Test connection
            self.redis.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError:
            logger.warning(f"Could not connect to Redis at {redis_host}:{redis_port}. Using in-memory storage instead.")
            self.redis = None
        
        # In-memory storage as fallback or for testing
        self.pdf_contents = {}
        
        # Create content directory if it doesn't exist
        self.content_dir = os.getenv("PDF_CONTENT_DIR", "pdf_contents")
        os.makedirs(self.content_dir, exist_ok=True)
    
    def _get_redis_key(self, content_id: str) -> str:
        """Get the Redis key for a PDF content"""
        return f"pdf:content:{content_id}"
    
    def _save_to_redis(self, content_id: str, data: Dict[str, Any]) -> bool:
        """Save PDF content to Redis"""
        if not self.redis:
            return False
        
        try:
            # Convert data to JSON string
            json_data = json.dumps(data)
            # Save to Redis
            self.redis.set(self._get_redis_key(content_id), json_data)
            return True
        except Exception as e:
            logger.error(f"Error saving to Redis: {str(e)}")
            return False
    
    def _load_from_redis(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Load PDF content from Redis"""
        if not self.redis:
            return None
        
        try:
            # Get data from Redis
            json_data = self.redis.get(self._get_redis_key(content_id))
            if not json_data:
                return None
            
            # Parse JSON data
            return json.loads(json_data)
        except Exception as e:
            logger.error(f"Error loading from Redis: {str(e)}")
            return None
    
    def _save_to_file(self, content_id: str, data: Dict[str, Any]) -> bool:
        """Save PDF content to a file"""
        try:
            # Generate filename
            filename = os.path.join(self.content_dir, f"{content_id}.json")
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving to file: {str(e)}")
            return False
    
    def _load_from_file(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Load PDF content from a file"""
        try:
            # Generate filename
            filename = os.path.join(self.content_dir, f"{content_id}.json")
            
            # Check if file exists
            if not os.path.exists(filename):
                return None
            
            # Load from file
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading from file: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text from a PDF file
        
        Args:
            pdf_bytes: The PDF file content as bytes
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            # Create a PDF reader object
            pdf_stream = io.BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_stream)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            text = ""
            pages = []
            
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                text += page_text + "\n\n"
                pages.append(page_text)
            
            # Return extracted text and metadata
            return {
                "content": text,
                "pages": pages,
                "page_count": num_pages
            }
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def process_pdf(self, content_id: str, filename: str, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Process a PDF document
        
        Args:
            content_id: Unique ID for the PDF content
            filename: Original filename
            pdf_bytes: The PDF file content as bytes
            
        Returns:
            Dict containing processed PDF content and metadata
        """
        try:
            # Extract text from PDF
            extracted_data = self.extract_text_from_pdf(pdf_bytes)
            
            # Create PDF content object
            pdf_content = {
                "filename": filename,
                "content": extracted_data["content"],
                "pages": extracted_data["pages"],
                "page_count": extracted_data["page_count"],
                "created_at": time.time()
            }
            
            # Save to Redis if available
            redis_success = self._save_to_redis(content_id, pdf_content)
            
            # Also save to file as backup
            file_success = self._save_to_file(content_id, pdf_content)
            
            # Store in memory
            self.pdf_contents[content_id] = pdf_content
            
            logger.info(f"Processed PDF '{filename}' with ID {content_id}: {extracted_data['page_count']} pages")
            
            return pdf_content
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise Exception(f"Error processing PDF: {str(e)}")
    
    async def process_pdf_background(self, content_id: str, filename: str, pdf_bytes: bytes):
        """Background task for processing PDFs"""
        try:
            self.process_pdf(content_id, filename, pdf_bytes)
        except Exception as e:
            logger.error(f"Error in background PDF processing: {str(e)}")
    
    def get_pdf_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get PDF content by ID
        
        Args:
            content_id: ID of the PDF content
            
        Returns:
            Dict containing PDF content and metadata, or None if not found
        """
        # Try to get from memory
        if content_id in self.pdf_contents:
            return self.pdf_contents[content_id]
        
        # Try to get from Redis
        redis_data = self._load_from_redis(content_id)
        if redis_data:
            # Cache in memory
            self.pdf_contents[content_id] = redis_data
            return redis_data
        
        # Try to get from file
        file_data = self._load_from_file(content_id)
        if file_data:
            # Cache in memory
            self.pdf_contents[content_id] = file_data
            return file_data
        
        # Not found
        return None
    
    def list_pdf_contents(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available PDF contents
        
        Returns:
            Dict mapping content IDs to PDF content metadata
        """
        # Start with in-memory contents
        contents = self.pdf_contents.copy()
        
        # If Redis is available, get keys
        if self.redis:
            try:
                # Get all keys matching pattern
                keys = self.redis.keys("pdf:content:*")
                
                # Load each content
                for key in keys:
                    content_id = key.split(":")[-1]
                    if content_id not in contents:
                        content = self._load_from_redis(content_id)
                        if content:
                            contents[content_id] = content
            except Exception as e:
                logger.error(f"Error listing Redis PDF contents: {str(e)}")
        
        # Check files as well
        try:
            for filename in os.listdir(self.content_dir):
                if filename.endswith(".json"):
                    content_id = filename[:-5]  # Remove .json extension
                    if content_id not in contents:
                        content = self._load_from_file(content_id)
                        if content:
                            contents[content_id] = content
        except Exception as e:
            logger.error(f"Error listing file PDF contents: {str(e)}")
        
        return contents