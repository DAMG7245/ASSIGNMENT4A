from fastapi import FastAPI, HTTPException
from fastapi import APIRouter

from pydantic import BaseModel
from pathlib import Path
import logging

from app.utils.enterprise.pdf_utlis import extract_pdf_content

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class PDFExtractionRequest(BaseModel):
    pdf_path: str
    output_dir: str


# @router.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
  
#     logger.info(f"Received file: {file.filename}")

#     # Validate file extension
#     if not file.filename.lower().endswith(".pdf"):
#         logger.error("Invalid file type uploaded.")
#         raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
#     # Define output directory
#     output_dir = Path("./data/parsed")
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Save the uploaded file
#     file_path = output_dir / file.filename
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#         logger.info(f"File saved successfully: {file_path}")
#     except Exception as e:
#         logger.error(f"Failed to save the file: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to save the file: {str(e)}")
    
#     # Process the PDF
#     try:
#         converter = DoclingConverter()
#         result = converter.process_pdf(file_path, output_dir)
        
#         # Validate extracted content
#         if not result or (isinstance(result, list) and not any(res.get("content") for res in result)):
#             logger.warning(f"No valid content extracted from: {file.filename}")
#             raise HTTPException(status_code=422, detail="The uploaded PDF is empty or not readable.")

#         logger.info(f"PDF processing completed: {file.filename}")
#     except Exception as e:
#         logger.error(f"Failed to process the PDF: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to process the PDF: {str(e)}")
    
#     return {
#         "filename": file.filename,
#         "status": "success",
#         "result": result
#     }



@router.post("/extract-pdf/enterprise")
async def extract_pdf(request: PDFExtractionRequest):
    try:
        # Validate paths
        pdf_path = Path(request.pdf_path)
        output_dir = Path(request.output_dir)
        
        # Check if PDF file exists
        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"PDF file not found at path: {pdf_path}"
            )
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract PDF content
        zip_path = extract_pdf_content(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir)
        )
        
        
        return {
            "status": "success",
            "zip_path": zip_path,
            "message": "PDF extraction completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

__all__ = ['router'] 