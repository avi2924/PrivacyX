from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import google.generativeai as genai

# === Load environment variables ===
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "vdpo_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === Configure Gemini ===
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

# === FastAPI setup ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Qdrant + Embedder ===
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# === Dummy in-memory user DB ===
VALID_USERS = {"admin": "admin123"}
sessions = {}

def get_current_user(request: Request):
    username = request.cookies.get("username")
    if not username or username not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return username

# === Login Page ===
@app.get("/", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in VALID_USERS and VALID_USERS[username] == password:
        sessions[username] = True
        response = RedirectResponse("/home", status_code=302)
        response.set_cookie("username", username)
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "❌ Invalid credentials. Please try again or sign up."
        })

# === Signup Page ===
@app.get("/signup", response_class=HTMLResponse)
def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})

@app.post("/signup", response_class=HTMLResponse)
def signup(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in VALID_USERS:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "⚠️ Username already exists."
        })
    
    VALID_USERS[username] = password
    sessions[username] = True
    response = RedirectResponse("/home", status_code=302)
    response.set_cookie("username", username)
    return response

# === Logout ===
@app.get("/logout")
def logout(request: Request):
    username = request.cookies.get("username")
    sessions.pop(username, None)
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("username")
    return response

# === Home Page ===
@app.get("/home", response_class=HTMLResponse)
def home(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("home.html", {"request": request, "user": user})

# === Ask Endpoint ===
@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...), user: str = Depends(get_current_user)):
    query_vector = embed_model.encode([question])[0]
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True
    )

    context_chunks, sources = [], []
    for i, r in enumerate(results, 1):
        context_chunks.append(f"{i}. {r.payload.get('text', '')}")
        sources.append(f"📄 {r.payload.get('source', '')} (score: {r.score:.4f})")

    prompt = f"""You are a Data Protection expert AI. Use the following document context to answer the user's question:\n\nCONTEXT:\n{chr(10).join(context_chunks)}\n\nQUESTION:\n{question}\n\nANSWER:"""

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error: {str(e)}"

    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user,
        "question": question,
        "answer": answer,
        "sources": sources
    })
