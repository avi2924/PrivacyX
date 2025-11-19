import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

# Base URL for EDPB documents
base_url = "https://edpb.europa.eu/our-work-tools/our-documents_en"
save_folder = "./"
os.makedirs(save_folder, exist_ok=True)

# Fetch HTML
response = requests.get(base_url)
soup = BeautifulSoup(response.content, "html.parser")

# Find all .pdf links that look like English versions
pdf_links = []
for link in soup.find_all("a", href=True):
    href = link["href"]
    if href.endswith(".pdf") and ("_en.pdf" in href or "EN.pdf" in href or "en.pdf" in href):
        full_url = urljoin(base_url, href)
        pdf_links.append(full_url)

print(f"🔎 Found {len(pdf_links)} English PDFs.")

# Download PDFs
for pdf_url in tqdm(pdf_links, desc="📥 Downloading English PDFs"):
    filename = os.path.basename(pdf_url)
    file_path = os.path.join(save_folder, filename)

    try:
        r = requests.get(pdf_url)
        with open(file_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"❌ Failed to download {pdf_url}: {e}")

print(f"\n✅ Downloaded {len(pdf_links)} English PDFs to: {save_folder}")
