import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
from tqdm import tqdm

BASE_URL = "https://www.privacyresources.eu"
GDPR_INDEX = BASE_URL + "/docs/gdpr/"
SAVE_DIR = "./gdpr_articles"
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Get chapter URLs
response = requests.get(GDPR_INDEX)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

chapter_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.startswith("/docs/gdpr/chapter"):
        full_url = urljoin(BASE_URL, href)
        chapter_links.append(full_url)

chapter_links = sorted(set(chapter_links))  # Remove duplicates
print(f"📘 Found {len(chapter_links)} chapters.")

# Step 2: Visit each chapter and extract text
for url in tqdm(chapter_links, desc="📖 Scraping chapters"):
    try:
        r = requests.get(url)
        r.raise_for_status()
        chapter_soup = BeautifulSoup(r.text, "html.parser")

        # First try extracting .entry divs
        entries = chapter_soup.find_all("div", class_="entry")
        content = ""

        if entries:
            for entry in entries:
                content += entry.get_text(separator="\n", strip=True) + "\n\n"
        else:
            # Fallback: try <main> or <article>
            fallback = chapter_soup.find("main") or chapter_soup.find("article")
            if fallback:
                content = fallback.get_text(separator="\n", strip=True)
            else:
                print(f"⚠️ No content found for {url}")

        chapter_name = url.rstrip("/").split("/")[-1]
        filename = os.path.join(SAVE_DIR, f"{chapter_name}.txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")


print("✅ GDPR chapter scraping complete.")
