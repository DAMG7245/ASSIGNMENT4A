from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
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

class PDFSummaryRequest(BaseModel):
    content_id: str = Field(..., description="ID of the PDF content to summarize")
    model: str = Field(None, description="LLM model to use for summarization")
    max_length: int = Field(500, description="Maximum length of the summary in tokens")

class PDFQuestionRequest(BaseModel):
    content_id: str = Field(..., description="ID of the PDF content")
    question: str = Field(..., description="Question about the PDF content")
    model: str = Field(None, description="LLM model to use for answering")
    max_tokens: int = Field(1000, description="Maximum length of the answer in tokens")

class PDFContentResponse(BaseModel):
    content_id: str = Field(..., description="ID of the PDF content")
    filename: str = Field(..., description="Original filename")
    content_preview: str = Field(..., description="Preview of the PDF content")
    page_count: int = Field(..., description="Number of pages in the PDF")
    status: str = Field("success", description="Status of the operation")

class PDFSummaryResponse(BaseModel):
    content_id: str = Field(..., description="ID of the PDF content")
    summary: str = Field(..., description="Generated summary")
    model: str = Field(..., description="Model used for summarization")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")

class PDFQuestionResponse(BaseModel):
    content_id: str = Field(..., description="ID of the PDF content")
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
    Select previously parsed PDF content by ID
    """
    try:
        content = pdf_controller.get_pdf_content(request.content_id)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDF content with ID {request.content_id} not found")
        
        return {
            "content_id": request.content_id,
            "filename": content.get("filename", "unknown"),
            "content_preview": content.get("content", "")[:200] + "...",
            "page_count": content.get("page_count", 0),
            "status": "success"
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
            background_tasks.add_task(
                pdf_controller.process_pdf_background,
                content_id,
                file.filename,
                pdf_bytes
            )
            processing_status = "processing"
        else:
            # Process immediately
            pdf_content = pdf_controller.process_pdf(
                content_id,
                file.filename,
                pdf_bytes
            )
            processing_status = "success"
        
        # Get the processed content
        content = pdf_controller.get_pdf_content(content_id)
        if not content and not background_tasks:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
        
        return {
            "content_id": content_id,
            "filename": file.filename,
            "content_preview": content.get("content", "Processing...")[:200] + "..." if content else "Processing...",
            "page_count": content.get("page_count", 0) if content else 0,
            "status": processing_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize/", response_model=PDFSummaryResponse)
async def summarize_pdf(request: PDFSummaryRequest):
    """
    Generate a summary of the PDF content
    """
    try:
        # Get the PDF content
        content = pdf_controller.get_pdf_content(request.content_id)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDF content with ID {request.content_id} not found")
        
        # Create prompt for summarization
        prompt = f"Please provide a concise summary of the following document in at most {request.max_length} tokens:\n\n{content['content']}"
        
        # Generate summary using LLM
        result = llm_controller.generate(
            prompt=prompt,
            model=request.model,
            max_tokens=request.max_length,
            temperature=0.5,  # Lower temperature for more factual summaries
        )
        
        return {
            "content_id": request.content_id,
            "summary": result["text"],
            "model": result["model"],
            "usage": result.get("usage")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask_question/", response_model=PDFQuestionResponse)
async def ask_question(request: PDFQuestionRequest):
    """
    Answer a question about the PDF content
    """
    try:
        # Get the PDF content
        content = pdf_controller.get_pdf_content(request.content_id)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDF content with ID {request.content_id} not found")
        
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
            "content_id": request.content_id,
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

@router.get("/list_contents/")
async def list_pdf_contents():
    """
    List all available PDF contents
    """
    try:
        contents = pdf_controller.list_pdf_contents()
        return {
            "total": len(contents),
            "contents": [
                {
                    "content_id": content_id,
                    "filename": content.get("filename", "unknown"),
                    "page_count": content.get("page_count", 0),
                    "created_at": content.get("created_at", None)
                }
                for content_id, content in contents.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error listing PDF contents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))