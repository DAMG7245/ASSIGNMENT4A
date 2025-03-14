import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import boto3
import uuid
from datetime import datetime
import re
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "parsing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PdfConverter:
    def __init__(self, bucket_name: str = "damg7245-datanexus-pro"):
        """Initialize PdfConverter with S3 bucket configuration"""
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.image_scale = 2.0
        self.dpi = 300

    def get_s3_url(self, s3_key: str) -> str:
        """Generate an S3 URL for a given key"""
        return f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"

    def upload_to_s3(self, local_path: Path, s3_key: str) -> bool:
        """Upload a file to S3 with error handling"""
        try:
            if not local_path.exists():
                logger.error(f"File not found: {local_path}")
                return False
                
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {str(e)}")
            return False

    def save_image_safely(self, img_data: bytes, image_path: Path, format: str = 'PNG') -> bool:
        """Safely save image data to file with error handling"""
        try:
            if not img_data:
                logger.error("No image data provided")
                return False
                
            img = Image.open(io.BytesIO(img_data))
            img.save(str(image_path), format=format)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            return False

    def detect_and_extract_tables(self, image_path: str) -> List[np.ndarray]:
        """Detect and extract tables from an image using OpenCV with error handling"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return []
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect lines
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 100:  # Filter small regions
                    table_region = image[y:y+h, x:x+w]
                    tables.append(table_region)
            
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []

    def extract_image_safely(self, pdf_document: fitz.Document, xref: int) -> Optional[Dict]:
        """Safely extract image from PDF with error handling"""
        try:
            image_info = pdf_document.extract_image(xref)
            if image_info and 'image' in image_info:
                return image_info
        except Exception as e:
            logger.error(f"Failed to extract image {xref}: {str(e)}")
        return None

    def process_and_upload_images(
        self,
        pdf_document: fitz.Document,
        images_dir: Path,
        base_filename: str,
        base_s3_path: str
    ) -> Tuple[List[Dict[str, str]], int, int]:
        """Process and upload images from PDF pages with robust error handling"""
        image_info = []
        table_count = 0
        image_count = 0
        
        try:
            # Ensure images directory exists
            images_dir.mkdir(parents=True, exist_ok=True)
            
            for page_num in range(len(pdf_document)):
                try:
                    page = pdf_document[page_num]
                    
                    # Convert page to image with error handling
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(self.image_scale, self.image_scale))
                        img_data = pix.tobytes()
                        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                        
                        page_image_path = images_dir / f"{base_filename}_page_{page_num+1}.png"
                        img.save(str(page_image_path))
                        
                        # Process tables
                        tables = self.detect_and_extract_tables(str(page_image_path))
                        
                        for table_img in tables:
                            table_count += 1
                            table_filename = f"{base_filename}_table_{table_count}.png"
                            table_path = images_dir / table_filename
                            
                            cv2.imwrite(str(table_path), table_img)
                            
                            table_s3_key = f"{base_s3_path}/images/tables/{table_filename}"
                            if self.upload_to_s3(table_path, table_s3_key):
                                image_info.append({
                                    'type': 'table',
                                    'local_path': str(table_path),
                                    's3_url': self.get_s3_url(table_s3_key)
                                })
                    
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}")
                        continue
                    
                    # Extract and process regular images
                    try:
                        for img in page.get_images():
                            image_count += 1
                            image_info_dict = self.extract_image_safely(pdf_document, img[0])
                            
                            if image_info_dict:
                                image_ext = image_info_dict.get("ext", "png")
                                image_filename = f"{base_filename}_image_{image_count}.{image_ext}"
                                image_path = images_dir / image_filename
                                
                                if self.save_image_safely(image_info_dict["image"], image_path):
                                    image_s3_key = f"{base_s3_path}/images/content/{image_filename}"
                                    if self.upload_to_s3(image_path, image_s3_key):
                                        image_info.append({
                                            'type': 'image',
                                            'local_path': str(image_path),
                                            's3_url': self.get_s3_url(image_s3_key)
                                        })
                    
                    except Exception as e:
                        logger.error(f"Error extracting images from page {page_num}: {str(e)}")
                        continue
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    continue
            
            return image_info, table_count, image_count
        
        except Exception as e:
            logger.error(f"Error in process_and_upload_images: {str(e)}")
            return [], 0, 0

    def extract_text_with_layout(self, pdf_document: fitz.Document) -> str:
        """Extract text while preserving layout with error handling"""
        try:
            markdown_content = []
            
            for page in pdf_document:
                try:
                    blocks = page.get_text("dict")["blocks"]
                    
                    for block in blocks:
                        if block["type"] == 0:  # Text
                            for line in block["lines"]:
                                try:
                                    text = " ".join(span["text"] for span in line["spans"])
                                    font_size = line["spans"][0]["size"]
                                    if font_size > 14:
                                        markdown_content.append(f"## {text}\n")
                                    else:
                                        markdown_content.append(f"{text}\n")
                                except Exception as e:
                                    logger.error(f"Error processing line: {str(e)}")
                                    continue
                        elif block["type"] == 1:  # Image
                            markdown_content.append("![image](placeholder)\n")
                
                except Exception as e:
                    logger.error(f"Error processing page text: {str(e)}")
                    continue
            
            return "\n".join(markdown_content)
        
        except Exception as e:
            logger.error(f"Error in extract_text_with_layout: {str(e)}")
            return ""

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file with comprehensive error handling"""
        try:
            # Create output directories
            output_dir = Path("output")
            markdown_dir = output_dir / "markdown"
            images_dir = output_dir / "images"
            
            for dir_path in [markdown_dir, images_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Create unique folder name for S3
            unique_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            process_folder = f"PDF_Extract_{timestamp}_{unique_id}"
            base_s3_path = f"pdf-extract/{process_folder}"

            # Open PDF with error handling
            try:
                pdf_document = fitz.open(pdf_path)
            except Exception as e:
                logger.error(f"Failed to open PDF: {str(e)}")
                return {'status': 'error', 'message': f"Failed to open PDF: {str(e)}"}

            try:
                base_filename = pdf_path.stem
                
                # Process images first
                image_info, table_count, image_count = self.process_and_upload_images(
                    pdf_document, images_dir, base_filename, base_s3_path
                )
                
                # Extract text
                markdown_content = self.extract_text_with_layout(pdf_document)
                
                # Update markdown content with image URLs
                for img in image_info:
                    markdown_content = markdown_content.replace(
                        "![image](placeholder)",
                        f"![{img['type']}]({img['s3_url']})",
                        1
                    )
                
                # Save and upload markdown
                markdown_path = markdown_dir / f"{base_filename}.md"
                try:
                    markdown_path.write_text(markdown_content)
                except Exception as e:
                    logger.error(f"Failed to save markdown: {str(e)}")
                    return {'status': 'error', 'message': f"Failed to save markdown: {str(e)}"}
                
                markdown_s3_key = f"{base_s3_path}/markdown/{base_filename}.md"
                if not self.upload_to_s3(markdown_path, markdown_s3_key):
                    return {'status': 'error', 'message': "Failed to upload markdown to S3"}
                
                return {
                    'status': 'success',
                    'markdown_path': str(markdown_path),
                    'markdown_content': markdown_content,
                    'markdown_s3_url': self.get_s3_url(markdown_s3_key),
                    's3_base_path': f"s3://{self.bucket_name}/{base_s3_path}",
                    'images': [{
                        'type': img['type'],
                        's3_url': img['s3_url']
                    } for img in image_info],
                    'table_count': table_count,
                    'image_count': image_count
                }
            
            finally:
                try:
                    pdf_document.close()
                except:
                    pass

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {'status': 'error', 'message': str(e)}