import requests
from pathlib import Path
import logging
from markdownify import markdownify
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

def fetch_markdown(url, output_dir, api_key):
    """Fetch Markdown content from a URL using Jina API and save it to a file."""
    base_url = "https://r.jina.ai/"
    jina_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/markdown"
    }
    
    params = {
        "x-respond-with": "markdown",  # Ensure Jina API returns Markdown
        "token_budget": 200000,
        "timeout": 10
    }

    try:
        # Request processed Markdown from Jina AI
        jina_response = requests.get(
            f"{base_url}{url}",
            headers=jina_headers,
            params=params
        )
        jina_response.raise_for_status()
        
        # Extract Markdown content
        markdown_content = jina_response.text
        
        # Save Markdown content to file
        md_path = Path(output_dir) / "website.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        _log.info(f"Markdown content fetched and saved to: {md_path}")
        
        return md_path, markdown_content

    except requests.exceptions.RequestException as e:
        _log.error(f"Failed to fetch {url}. Error: {str(e)}")
        raise ValueError(f"Failed to fetch {url}. Error: {str(e)}")

def extract_image_name(url):
    """Extract image filename from URL."""
    pattern = r"(?<=/)([^/]+\.(?:jpg|jpeg|png|svg|gif))$"
    match = re.search(pattern, url, re.IGNORECASE)
    return match.group(1) if match else None

def download_and_replace_images(md_content, output_dir):
    """Download images (including SVG) and replace URLs in Markdown content with local paths."""
    pattern = r'!\[.*?\]\((https?://[^\s]+)\)'
    matches = re.findall(pattern, md_content)
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://www.google.com",
    }
    
    session = requests.Session()  # Use a session for persistent headers
    
    for img_url in matches:
        image_name = extract_image_name(img_url)
        if image_name:
            local_path = images_dir / image_name
            try:
                response = session.get(img_url, headers=headers, stream=True, allow_redirects=True)
                response.raise_for_status()
                
                with open(local_path, "wb") as img_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            img_file.write(chunk)
                
                md_content = md_content.replace(img_url, str(local_path))
                _log.info(f"Downloaded and replaced: {img_url} -> {local_path}")
            except requests.exceptions.HTTPError as http_err:
                _log.warning(f"HTTP error while downloading image {img_url}: {http_err}")
            except requests.exceptions.RequestException as req_err:
                _log.warning(f"Request error while downloading image {img_url}: {req_err}")
    
    md_path = Path(output_dir) / "website.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    _log.info(f"Updated Markdown content saved to: {md_path}")

def main():
    # Configuration
    url = "https://sreenidhi.edu.in/"
    output_dir = "test_output"
    api_key = "jina_ad2a179a517a40f991ba1a2de1a1f925YS64rXRYkejV3OIHif7D7sA-U0R1"

    try:
        # Fetch Markdown content
        md_path, md_content = fetch_markdown(url, output_dir, api_key)
        
        # Download images and replace URLs with local paths
        download_and_replace_images(md_content, output_dir)
        
        _log.info("Markdown content successfully processed and saved.")
    except Exception as e:
        _log.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
