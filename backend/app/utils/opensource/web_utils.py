import requests
import base64
import logging
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import re
from markdownify import markdownify

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def find_image_urls(self, text):
        """Find all image URLs in text content."""
        pattern = r'https?://[^\s<>"]+?\.(?:jpg|jpeg|png|svg|gif)(?=\s|$)'
        matches = re.findall(pattern, text, re.IGNORECASE) or []
        
        soup = BeautifulSoup(text, "html.parser")
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url and "images/image_" not in img_url:
                matches.append(img_url)
        
        return matches

    def extract_image_name(self, url):
        """Extract image filename from URL."""
        pattern = r"(?<=/)([^/]+\.(?:jpg|jpeg|png|svg|gif))$"
        match = re.search(pattern, url, re.IGNORECASE)
        return match.group(1) if match else None

    def is_data_url(self, url):
        """Check if URL is a data URL."""
        return url.startswith('data:')

    def save_data_url(self, data_url, output_path):
        """Save data URL content to file."""
        try:
            header, data = data_url.split(',', 1)
            file_data = base64.b64decode(data) if 'base64' in header else data.encode('utf-8')
            
            with open(output_path, 'wb') as f:
                f.write(file_data)
            return True
        except Exception as e:
            _log.warning(f"Failed to save data URL: {str(e)}")
            return False

    def fetch_html(self, url):
        """Fetch HTML content from URL."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            html_content = response.text

            raw_html_path = self.output_dir / "website.html"
            raw_html_path.parent.mkdir(parents=True, exist_ok=True)

            with open(raw_html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            _log.info(f"HTML content fetched and saved to: {raw_html_path}")
            return raw_html_path, html_content

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch {url}. Error: {str(e)}")

    def extract_images(self, html_content, base_url):
        """Extract and download images from HTML content."""
        soup = BeautifulSoup(html_content, "html.parser")
        image_counter = 0
        
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if not img_url:
                continue

            try:
                if self.is_data_url(img_url):
                    image_counter = self._handle_data_url_image(img_url, image_counter, html_content)
                else:
                    image_counter = self._handle_regular_image(img_url, base_url, image_counter)
            except Exception as e:
                _log.warning(f"Failed to download image: {img_url}. Error: {e}")

        _log.info(f"{'No' if image_counter == 0 else image_counter} images {'found' if image_counter == 0 else 'downloaded'}.")

    def _handle_data_url_image(self, img_url, counter, html_content):
        """Handle data URL images."""
        image_extension = ".svg" if "svg+xml" in img_url else ".png"
        image_name = self.extract_image_name(img_url) or f"image_{counter}{image_extension}"
        image_filename = self.images_dir / image_name
        
        raw_html_path = self.output_dir / "website.html"
        with open(raw_html_path, "w", encoding="utf-8") as f:
            replaced_content = html_content.replace(img_url, f"/images/{image_name}")
            f.write(replaced_content)
        
        if self.save_data_url(img_url, image_filename):
            _log.info(f"Data URL image saved: {image_filename}")
            return counter + 1
        return counter

    def _handle_regular_image(self, img_url, base_url, counter):
        """Handle regular image URLs."""
        if not img_url.startswith(("http://", "https://")):
            img_url = urljoin(base_url, img_url)
        
        headers = {**self.headers, "Accept": "image/webp,image/apng,image/*,*/*;q=0.8", "Referer": base_url}
        response = requests.get(img_url, headers=headers, stream=True)
        response.raise_for_status()

        image_name = self.extract_image_name(img_url)
        if not image_name:
            extension = response.headers.get('content-type', '').split('/')[-1].split(';')[0] or 'jpg'
            image_name = f"image_{counter}.{extension}"
        
        image_filename = self.images_dir / image_name
        with open(image_filename, "wb") as img_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    img_file.write(chunk)
        
        _log.info(f"Image saved: {image_filename}")
        return counter + 1

    def replace_image_urls(self, html_content):
        """Replace image URLs with local paths."""
        try:
            list_urls = self.find_image_urls(html_content)
            for img_url in list_urls:
                image_name = self.extract_image_name(img_url)
                if image_name:
                    local_path = f"{self.output_dir}/images/{image_name}"
                    html_content = html_content.replace(img_url, local_path)
            
            raw_html_path = self.output_dir / "website.html"
            with open(raw_html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return True
        except Exception as e:
            _log.warning(f"Failed to replace image URLs: {str(e)}")
            return False

    def process_with_docling(self, input_html_path):
        """Process HTML with Docling and save as Markdown."""
        with open(input_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        doc_converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
        result = doc_converter.convert(input_html_path)

        if not result:
            _log.error("Failed to process the HTML file with Docling.")
            return False

        markdown_path = self.output_dir / "website_content.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            markdown = markdownify(html_content)
            markdown = markdown.replace("images/", f"{self.output_dir}/images/")
            f.write(markdown)
        
        _log.info(f"Processed content saved as Markdown: {markdown_path}")
        return True

def main():
    url = "https://sreenidhi.edu.in/"
    output_dir = "test_output"
    
    try:
        scraper = WebScraper(output_dir)
        
        # Fetch HTML content
        html_path, html_content = scraper.fetch_html(url)
        
        # Extract and process images
        scraper.extract_images(html_content, url)
        
        # Update HTML with local image paths
        with open(html_path, "r", encoding="utf-8") as f:
            updated_html = f.read()
        scraper.replace_image_urls(updated_html)
        
        # Process with Docling
        scraper.process_with_docling(html_path)
        
        _log.info("HTML content successfully processed and saved.")
    except Exception as e:
        _log.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()