import requests

GDPR_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
filename = "gdpr_official_en.pdf"

response = requests.get(GDPR_URL)
response.raise_for_status()

with open(filename, "wb") as f:
    f.write(response.content)

print("✅ GDPR Regulation downloaded as:", filename)
