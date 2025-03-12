# from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, APIRouter
# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any
# from pathlib import Path
# import logging
# import time
# import json
# import boto3
# from botocore.config import Config

# from ..utils.redis_client import get_redis_connection
# from ..utils.enterprise.pdf_handler_utils import process_zip
# from llm_service.llm_manager import LLMManager

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# router = APIRouter()
# llm_manager = LLMManager()

# # ZIP Processing models
# class ZIPProcessRequest(BaseModel):
#     zip_path: str

# # LLM Request models
# class LLMRequest(BaseModel):
#     prompt: str
#     model: str
#     document_content: Optional[str] = None
#     document_id: Optional[str] = None
#     operation_type: str = "chat"
#     conversation_id: Optional[str] = None

# # S3 Utilities
# def generate_presigned_url(bucket_name: str, object_key: str, expiration: int = 3600):
#     """Generate a presigned URL for an S3 object"""
#     s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
#     try:
#         url = s3_client.generate_presigned_url(
#             'get_object',
#             Params={
#                 'Bucket': bucket_name,
#                 'Key': object_key
#             },
#             ExpiresIn=expiration
#         )
#         return url
#     except Exception as e:
#         logger.error(f"Error generating presigned URL: {str(e)}")
#         raise

# # ZIP Processing Endpoints
# @router.post("/process-zip/enterprise")
# async def process_zip_file(request: ZIPProcessRequest):
#     try:
#         # Validate ZIP file path
#         zip_path = Path(request.zip_path)
#         if not zip_path.exists():
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"ZIP file not found at path: {zip_path}"
#             )
        
#         if not zip_path.suffix.lower() == '.zip':
#             raise HTTPException(
#                 status_code=400,
#                 detail="File must be a ZIP archive"
#             )
        
#         # Process the ZIP file and get the S3 base path
#         base_url = process_zip(zip_path=str(zip_path))
#         print("Output directory:", base_url)
        
#         # Extract bucket name and key prefix from the base_url
#         # Convert https://bucket.s3.amazonaws.com/path to bucket and path
#         bucket_name = "damg7245-datanexus-pro"
#         base_path = base_url.split(f"{bucket_name}.s3.amazonaws.com/")[1]
        
#         # Generate presigned URLs
#         markdown_key = f"{base_path}markdown/content.md"
#         images_key = f"{base_path}images"
        
#         markdown_url = generate_presigned_url(bucket_name, markdown_key)
        
#         return {
#             "status": "success",
#             "message": "ZIP file processed successfully",
#             "output_locations": {
#                 "markdown_file": markdown_url,
#                 "base_path": base_path,
#                 "bucket": bucket_name
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing ZIP file: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing ZIP file: {str(e)}"
#         )

# # LLM Endpoints
# @router.get("/llm/models")
# async def get_available_models():
#     """Get list of available LLM models"""
#     try:
#         models = llm_manager.get_available_models()
#         return {"models": models}
#     except Exception as e:
#         logger.error(f"Error fetching models: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/llm/generate")
# async def generate_text(request: LLMRequest):
#     """Generic endpoint for text generation with LLM"""
#     try:
#         result = llm_manager.generate_text(
#             model_id=request.model,
#             prompt=request.prompt,
#             document_content=request.document_content,
#             operation_type=request.operation_type
#         )
#         return result
#     except Exception as e:
#         logger.error(f"Error generating text: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/llm/ask_question")
# async def ask_question(request: LLMRequest):
#     """Answer a question about document content"""
#     try:
#         # If document_id is provided, fetch content from Redis
#         document_content = request.document_content
#         if request.document_id and not document_content:
#             redis = get_redis_connection()
#             doc_key = f"document:{request.document_id}"
#             doc_data = redis.get(doc_key)
#             if doc_data:
#                 document = json.loads(doc_data)
#                 document_content = document.get("content", "")
        
#         result = llm_manager.generate_text(
#             model_id=request.model,
#             prompt=request.prompt,
#             document_content=document_content,
#             operation_type="chat"
#         )
#         return result
#     except Exception as e:
#         logger.error(f"Error answering question: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/llm/summarize")
# async def summarize_document(request: LLMRequest):
#     """Generate a summary of document content"""
#     try:
#         # If document_id is provided, fetch content from Redis
#         document_content = request.document_content
#         if request.document_id and not document_content:
#             redis = get_redis_connection()
#             doc_key = f"document:{request.document_id}"
#             doc_data = redis.get(doc_key)
#             if doc_data:
#                 document = json.loads(doc_data)
#                 document_content = document.get("content", "")
        
#         result = llm_manager.generate_text(
#             model_id=request.model,
#             prompt="Provide a comprehensive summary of this document.",
#             document_content=document_content,
#             operation_type="summarize"
#         )
#         return result
#     except Exception as e:
#         logger.error(f"Error summarizing document: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/llm/extract_keypoints")
# async def extract_keypoints(request: LLMRequest):
#     """Extract key points from document content"""
#     try:
#         # If document_id is provided, fetch content from Redis
#         document_content = request.document_content
#         if request.document_id and not document_content:
#             redis = get_redis_connection()
#             doc_key = f"document:{request.document_id}"
#             doc_data = redis.get(doc_key)
#             if doc_data:
#                 document = json.loads(doc_data)
#                 document_content = document.get("content", "")
        
#         result = llm_manager.generate_text(
#             model_id=request.model,
#             prompt="Extract the key points from this document.",
#             document_content=document_content,
#             operation_type="extract_keypoints"
#         )
#         return result
#     except Exception as e:
#         logger.error(f"Error extracting key points: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))













from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from app.utils.enterprise.pdf_handler_utils import process_zip
from fastapi import APIRouter
import boto3
from botocore.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class ZIPProcessRequest(BaseModel):
    zip_path: str

def generate_presigned_url(bucket_name: str, object_key: str, expiration: int = 3600):
    """Generate a presigned URL for an S3 object"""
    s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key
            },
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise

@router.post("/process-zip/enterprise")
async def process_zip_file(request: ZIPProcessRequest):
    try:
        # Validate ZIP file path
        zip_path = Path(request.zip_path)
        if not zip_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"ZIP file not found at path: {zip_path}"
            )
        
        if not zip_path.suffix.lower() == '.zip':
            raise HTTPException(
                status_code=400,
                detail="File must be a ZIP archive"
            )
        
        # Process the ZIP file and get the S3 base path
        base_url = process_zip(zip_path=str(zip_path))
        print("Output directory:", base_url)
        
        # Extract bucket name and key prefix from the base_url
        # Convert https://bucket.s3.amazonaws.com/path to bucket and path
        bucket_name = "damg7245-datanexus-pro"
        base_path = base_url.split(f"{bucket_name}.s3.amazonaws.com/")[1]
        
        # Generate presigned URLs
        markdown_key = f"{base_path}markdown/content.md"
        images_key = f"{base_path}images"
        
        markdown_url = generate_presigned_url(bucket_name, markdown_key)
        
        return {
            "status": "success",
            "message": "ZIP file processed successfully",
            "output_locations": {
                "markdown_file": markdown_url,
                "base_path": base_path,
                "bucket": bucket_name
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing ZIP file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing ZIP file: {str(e)}"
        )