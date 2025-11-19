from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import os, requests

# Setup
START_URL = "https://www.privacyresources.eu/docs/edps/"
DOWNLOAD_DIR = "edps_pdfs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Configure Selenium
options = Options()
options.add_argument('--headless=new')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
driver = webdriver.Chrome(options=options)

try:
    print("🔍 Opening EDPS source page...")
    driver.get(START_URL)

    # Wait for EDPS links to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href*='edps.europa.eu']"))
    )

    # Collect all EDPS opinion page links
    links = [
        a.get_attribute("href")
        for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='edps.europa.eu']")
        if "publications/opinions" in a.get_attribute("href")
    ]

    print(f"📄 Found {len(links)} EDPS opinion pages.\n")

    pdf_links = []

    for url in tqdm(links, desc="🔍 Visiting EDPS pages"):
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
            )
            anchors = driver.find_elements(By.TAG_NAME, "a")
            found_any = False
            for a in anchors:
                href = a.get_attribute("href")
                if href and href.lower().endswith(".pdf"):
                    pdf_links.append(href)
                    found_any = True
            if not found_any:
                print(f"⚠️ No PDF link found on: {url}")
        except Exception as e:
            print(f"⚠️ Failed to extract PDF from {url} — {e}")

    print(f"\n📥 Total PDFs found: {len(pdf_links)}")
    
    # ✅ Download PDFs
    for pdf_url in tqdm(pdf_links, desc="📥 Downloading PDFs"):
        try:
            filename = os.path.join(DOWNLOAD_DIR, os.path.basename(pdf_url))
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"❌ Error downloading {pdf_url} — {e}")

    print("✅ All EDPS PDFs downloaded successfully.")

finally:
    driver.quit()
