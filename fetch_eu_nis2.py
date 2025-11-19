from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# Set up
URL = "https://digital-strategy.ec.europa.eu/en/policies/nis2-directive"
OUTPUT_FILE = "nis2_full_text.txt"

# Headless browser
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)
print(f"🌐 Visiting: {URL}")
driver.get(URL)
time.sleep(5)  # wait longer to allow JavaScript content

# Parse HTML
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# Extract all visible text inside <main> content
main_content = soup.find("main") or soup.find("body")

if not main_content:
    print("❌ Could not find main or body content.")
    exit()

# Collect all paragraph-level text
paragraphs = main_content.find_all(["p", "h1", "h2", "h3", "li"])

visible_text = ""
for tag in paragraphs:
    line = tag.get_text(strip=True)
    if line:
        visible_text += line + "\n\n"

# Save the output
if visible_text:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(visible_text)
    print(f"✅ Saved {len(visible_text.split())} words to {OUTPUT_FILE}")
else:
    print("⚠️ Page loaded, but no readable text found.")
