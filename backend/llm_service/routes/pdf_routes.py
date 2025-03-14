from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import os
from uuid import uuid4

from controller.pdf_controller import PDFController
from controller.llm_controller import LLMController

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/pdf",
    tags=["PDF Processing"],
    responses={404: {"description": "Not found"}},
)

# Request and response models
class PDFContentRequest(BaseModel):
    content_id: str = Field(..., description="ID of the previously parsed PDF content")
    folder_path: Optional[str] = Field(None, description="Path to the folder containing the PDF content")

class PDFSummaryRequest(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing the PDF content")
    model: str = Field(None, description="LLM model to use for summarization")
    max_length: int = Field(500, description="Maximum length of the summary in tokens")

class PDFQuestionRequest(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing the PDF content")
    question: str = Field(..., description="Question about the PDF content")
    model: str = Field(None, description="LLM model to use for answering")
    max_tokens: int = Field(1000, description="Maximum length of the answer in tokens")
class ContentSummaryRequest(BaseModel):
    content: str
    model: Optional[str] = None
    max_length: int = 1000
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SummaryResponse(BaseModel):
    summary: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
class PDFContentResponse(BaseModel):
    content_id: str = Field(..., description="ID of the PDF content")
    folder_path: Optional[str] = Field(None, description="Path to the folder containing the PDF content")
    filename: str = Field(..., description="Original filename")
    folder_name: str = Field("Unknown", description="Name of the folder containing the PDF")
    content_preview: str = Field(..., description="Preview of the PDF content")
    page_count: int = Field(..., description="Number of pages in the PDF")
    has_markdown: bool = Field(False, description="Whether markdown content is available")
    markdown_content: Optional[str] = Field(None, description="Markdown content if available")
    status: str = Field("success", description="Status of the operation")
    
    class Config:
        orm_mode = True

class PDFSummaryResponse(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing the PDF content")
    summary: str = Field(..., description="Generated summary")
    model: str = Field(..., description="Model used for summarization")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")

class PDFQuestionResponse(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing the PDF content")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    model: str = Field(..., description="Model used for answering")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")

# Initialize controllers
pdf_controller = PDFController()
llm_controller = LLMController()

@router.post("/select_pdfcontent/", response_model=PDFContentResponse)
async def select_pdf_content(request: PDFContentRequest):
    """
    Select previously parsed PDF content by ID and folder path
    """
    try:
        # Get the PDF content as a string
        content = pdf_controller.get_pdf_content(request.content_id, request.folder_path)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDF content with ID {request.content_id} not found")
        
        # Define other fields that might not be directly available from `content`
        filename = "unknown"  # You might want to retrieve this from the content or set a placeholder.
        folder_name = request.folder_path.split("/")[-1] if request.folder_path else "Unknown"
        page_count = 1  # Set to 1 if you're not calculating page count dynamically. You might retrieve this if needed.
        has_markdown = False  # Assuming no markdown content unless you fetch it

        # Return the content as part of the response
        return {
            "content_id": request.content_id,
            "folder_path": request.folder_path,
            "filename": filename,
            "folder_name": folder_name,
            "content_preview": content[:200] + "..." if content else "",
            "page_count": page_count,
            "has_markdown": has_markdown,
            "status": "success",
            "markdown_content":content
        }
    
    except Exception as e:
        logger.error(f"Error selecting PDF content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_pdf/", response_model=PDFContentResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a new PDF document
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
        
        # Generate a unique ID for this PDF content
        content_id = str(uuid4())
        
        # Read the PDF file
        pdf_bytes = await file.read()
        
        # Process the PDF in the background if background_tasks is provided
        if background_tasks:
            folder_path = background_tasks.add_task(
                pdf_controller.process_pdf_background,
                content_id,
                file.filename,
                pdf_bytes
            )
            processing_status = "processing"
        else:
            # Process immediately
            folder_path = pdf_controller.process_pdf(
                content_id,
                file.filename,
                pdf_bytes
            )
            processing_status = "success"
        
        # Get the processed content
        content = pdf_controller.get_pdf_content(content_id, folder_path)
        if not content and not background_tasks:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
        
        return {
            "content_id": content_id,
            "folder_path": folder_path,
            "filename": file.filename,
            "folder_name": content.get("folder_name", "Unknown") if content else "Processing...",
            "content_preview": content.get("content", "Processing...")[:200] + "..." if content else "Processing...",
            "page_count": content.get("page_count", 0) if content else 0,
            "has_markdown": content.get("has_markdown", False) if content else False,
            "status": processing_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize/", response_model=SummaryResponse)
async def summarize_pdf_content(
    request: dict,  # Use a dict to accept any combination of fields
    llm_controller: LLMController = Depends(lambda: LLMController())
):
    """
    Generate a summary using content_id or folder_path to retrieve content
    """
    try:
        content_text = None
        content_id = request.get("content_id")
        folder_path = request.get("folder_path")
        
        logger.info(f"Request received - content_id: {content_id}, folder_path: {folder_path}")
        
        # First check if content is provided directly in the request
        content_text = request.get("content")
        if not content_text and "markdown_content" in request:
            content_text = request.get("markdown_content")
        
        # If content is not in the request and content_id is provided, 
        # try to get it from Redis directly
        if not content_text and content_id:
            try:
                from utils.redis_utils import get_redis_connection
                redis_client = get_redis_connection()
                
                # Try to get content from Redis with the key format "markdown:{content_id}"
                redis_key = f"markdown:{content_id}"
                redis_data = redis_client.get(redis_key)
                
                if redis_data:
                    import json
                    data_dict = json.loads(redis_data)
                    content_text = data_dict.get("content")
                    logger.info(f"Content retrieved from Redis for ID {content_id}: {bool(content_text)}")
            except Exception as e:
                logger.error(f"Error retrieving content from Redis: {str(e)}")
        
        # If we still don't have content and we have a folder_path, 
        # we could try to download the file (implementation dependent on your system)
        if not content_text and folder_path:
            # Placeholder for your file download logic
            # This depends on how your system handles file downloads
            logger.warning(f"Content not found in Redis, folder_path provided but no handler implemented")
        
        # If we don't have content, raise an error
        if not content_text:
            raise HTTPException(
                status_code=400, 
                detail="Could not retrieve content. Provide content directly or a valid content_id."
            )
        
        # Extract other parameters
        model = request.get("model")
        max_length = request.get("max_length", 1000)
        temperature = request.get("temperature", 0.3)
        metadata = request.get("metadata", {})
        
        # Create a general purpose prompt
        prompt = f"""Provide a comprehensive yet concise summary of this document in {max_length} tokens.
        Extract the main points, key arguments, and essential conclusions.
        
        Document: {content_text}"""
        
        # Generate summary using LLM
        result = llm_controller.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_length,
            temperature=temperature,
        )
        
        # Post-process the summary
        summary = _format_summary(result["text"])
        
        # Build metadata
        if not metadata:
            metadata = {}
            
        if folder_path:
            metadata["folder_path"] = folder_path
        
        if content_id:
            metadata["content_id"] = content_id
            
        if "title" in request:
            metadata["title"] = request["title"]
        
        return {
            "summary": summary,
            "model": result["model"],
            "usage": result.get("usage"),
            "metadata": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
def _format_summary(text: str) -> str:
    """
    Format and clean up the generated summary
    """
    # Remove any redundant phrases that models sometimes add
    cleanup_phrases = [
        "Here is a summary of the document:",
        "Summary:",
        "In summary:",
        "To summarize:"
    ]
    
    result = text
    for phrase in cleanup_phrases:
        if result.startswith(phrase):
            result = result[len(phrase):].strip()
    
    # Ensure the summary doesn't end mid-sentence
    if result and result[-1] not in ['.', '!', '?']:
        # Find the last sentence end
        last_period = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
        if last_period > len(result) * 0.75:  # Only truncate if we don't lose too much content
            result = result[:last_period+1]
    
    return result

@router.post("/ask_question/", response_model=PDFQuestionResponse)
async def ask_question(request: PDFQuestionRequest):
    """
    Answer a question about the PDF content
    """
    try:
        # Get the PDF content using folder path
        content = pdf_controller.get_pdf_content_by_path(request.folder_path)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDF content with folder path {request.folder_path} not found")
        
        # Create prompt for question answering
        prompt = f"""Please answer the following question about this document:

Document content:
{content['content']}

Question: {request.question}

Provide a detailed answer based only on the information in the document. If the information to answer the question is not in the document, state that clearly."""
        
        # Generate answer using LLM
        result = llm_controller.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about documents based solely on their content."},
                {"role": "user", "content": prompt}
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=0.3,  # Lower temperature for more factual answers
        )
        
        return {
            "folder_path": request.folder_path,
            "question": request.question,
            "answer": result["text"],
            "model": result["model"],
            "usage": result.get("usage")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_all_pdfs/")
async def list_all_pdfs():
    """
    List all available PDF contents
    """
    try:
        # Simply call the controller and return its response
        contents = pdf_controller.list_all_pdfs()
        return contents
        
    except Exception as e:
        logger.error(f"Error listing PDF contents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))