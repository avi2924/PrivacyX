from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin

# Config
BASE_URL = "https://iapp.org/news/"
SAVE_DIR = "./iapp_articles"
os.makedirs(SAVE_DIR, exist_ok=True)

# Setup headless browser
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

# Load the news page
driver.get(BASE_URL)
time.sleep(5)  # Wait for JS content

# Parse page content
soup = BeautifulSoup(driver.page_source, "html.parser")
article_links = set()

# Collect article links
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "/news/a/" in href:
        full_url = urljoin(BASE_URL, href)
        article_links.add(full_url)

print(f"📰 Found {len(article_links)} articles.")

# Visit and save content
for link in article_links:
    print(f"📥 Visiting: {link}")
    try:
        driver.get(link)
        time.sleep(2)

        sub_soup = BeautifulSoup(driver.page_source, "html.parser")

        # Try multiple content containers
        content = (
            sub_soup.find("div", class_="content")
            or sub_soup.find("div", class_="article-content")
            or sub_soup.find("main")
        )

        if content:
            text = content.get_text(separator="\n", strip=True)
            slug = link.rstrip("/").split("/")[-1]
            filename = f"{slug}.txt"

            with open(os.path.join(SAVE_DIR, filename), "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Saved: {filename}")
        else:
            print("⚠️ Could not extract content from:", link)

    except Exception as e:
        print("❌ Error visiting:", link, e)

driver.quit()
print("✅ All articles processed and saved.")
