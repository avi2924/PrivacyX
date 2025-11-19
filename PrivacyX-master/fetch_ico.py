from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
import requests
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://ico.org.uk"
START_PAGE = "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/"
DOWNLOAD_DIR = "./"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Set up headless browser
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

# Get all guidance page links from the main page
driver.get(START_PAGE)
soup = BeautifulSoup(driver.page_source, "html.parser")

subpage_urls = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "/for-organisations/" in href and not href.endswith(".pdf"):
        full_url = href if href.startswith("http") else urljoin(BASE_URL, href)
        subpage_urls.append(full_url)

subpage_urls = list(set(subpage_urls))  # Remove duplicates
print(f"🔍 Found {len(subpage_urls)} subpages.")

# Visit each subpage and collect PDF links
pdf_links = set()

for url in tqdm(subpage_urls, desc="🔍 Crawling subpages"):
    try:
        driver.get(url)
        sub_soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in sub_soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                full_pdf_url = href if href.startswith("http") else urljoin(BASE_URL, href)
                pdf_links.add(full_pdf_url)
    except Exception as e:
        print("Error loading:", url, e)

driver.quit()
pdf_links = list(pdf_links)
print(f"📄 Found {len(pdf_links)} PDF files.")

# Download the PDFs
for pdf_url in tqdm(pdf_links, desc="📥 Downloading PDFs"):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        filename = os.path.join(DOWNLOAD_DIR, os.path.basename(pdf_url))
        with open(filename, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print("❌ Error downloading:", pdf_url, e)

print("✅ All ICO PDFs downloaded successfully.")
