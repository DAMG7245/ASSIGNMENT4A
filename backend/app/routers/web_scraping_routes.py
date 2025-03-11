import os
import logging
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any

# Import your existing scraping function
from app.utils.enterprise.web_utils import scrape_data, save_to_md
from app.utils.opensource.web_utils import fetch_html, extract_images_from_html, replace_html, process_html_with_docling;

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI router
router = APIRouter()

class WebScrapingRequest(BaseModel):
    url: str
    output_dir: str

@router.post("/web-scraping/enterprise")
async def extract_web_data(request: WebScrapingRequest) -> Dict[str, Any]:
    try:
        # Validate output directory
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Perform web scraping
        scraped_content = scrape_data(request.url)
       

        # Save scraped content to Markdown
        output_md_path = output_dir / "scraped_data.md"
        save_to_md(scraped_content, str(output_md_path))

        return {
            "status": "success",
            "saved_path": str(output_md_path),
            "message": "Web scraping completed successfully"
        }

    except Exception as e:
        logger.error(f"Error processing web scraping: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing web scraping: {str(e)}"


        )
    
class WebScrapingRequest(BaseModel):
    url: str
    output_dir: str

@router.post("/web-scraping/opensource")
async def extract_web_data(request: WebScrapingRequest):
    try:
        # Validate output directory
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch HTML
        html_path, html_content = fetch_html(request.url, output_dir)

        # Extract and download images
        extract_images_from_html(html_content, output_dir, request.url)

        # Replace image paths in HTML
        with open(html_path, "r", encoding="utf-8") as f:
            updated_html=f.read()
        replace_html(updated_html, output_dir)

        # Convert HTML to Markdown
        process_html_with_docling(html_path, output_dir)

        return {
            "status": "success",
            "saved_path": {
                "html": str(html_path),
                "markdown": str(output_dir / "website_content.md"),
                "images_folder": str(output_dir / "images"),
            },
            "message": "Web scraping completed successfully"
        }

    except Exception as e:
        logger.error(f"Error processing web scraping: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing web scraping: {str(e)}"
        )


# Export router
__all__ = ['router']
