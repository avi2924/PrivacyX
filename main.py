from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import google.generativeai as genai
import numpy as np

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
    # create embedding (ensure convert to plain python list)
    emb = embed_model.encode([question])[0]
    if isinstance(emb, np.ndarray):
        query_vector = emb.tolist()
    else:
        # some backends return list already — still ensure it's a plain list of floats
        query_vector = list(map(float, emb))

    # Use query_points (modern Qdrant Client). Some older clients had `search` or `search_points`.
    results = None
    try:
        # preferred modern API
        query_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            with_payload=True
        )
        # query_response typically contains .points (or .result depending on client version)
        if hasattr(query_response, "points"):
            hits = query_response.points
        elif hasattr(query_response, "result"):
            hits = query_response.result
        else:
            # attempt to treat as iterable if shape differs
            hits = list(query_response)
    except AttributeError:
        # fallback for older qdrant-client versions that might expose `search`
        if hasattr(qdrant, "search"):
            hits = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=5,
                with_payload=True
            )
        else:
            # re-raise with a helpful message if neither method exists
            raise RuntimeError("Your installed qdrant-client doesn't expose `query_points` or `search`. "
                               "Try upgrading/downgrading qdrant-client or check docs.") from None
    except Exception as e:
        # any other runtime error — surface to template
        return templates.TemplateResponse("home.html", {
            "request": request,
            "user": user,
            "question": question,
            "answer": f"Error querying vector DB: {e}",
            "sources": []
        })

    # Build context and sources from hits (payload structure assumed: payload['text'], payload['source'])
    context_chunks, sources = [], []
    for i, r in enumerate(hits, 1):
        payload = getattr(r, "payload", None) or (r.get("payload") if isinstance(r, dict) else {})
        score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else None)
        text = payload.get("text", "") if payload else ""
        src = payload.get("source", "") if payload else ""
        context_chunks.append(f"{i}. {text}")
        if score is not None:
            sources.append(f"📄 {src} (score: {float(score):.4f})")
        else:
            sources.append(f"📄 {src}")

    prompt = f"""You are a Data Protection expert AI. Use the following document context to answer the user's question:\n\nCONTEXT:\n{chr(10).join(context_chunks)}\n\nQUESTION:\n{question}\n\nANSWER:"""

    try:
        response = gemini_model.generate_content(prompt)
        # Be robust to different shapes of Gemini output
        if hasattr(response, "text") and response.text:
            answer = response.text.strip()
        elif hasattr(response, "candidates"):
            # candidates -> list of candidate objects with 'content' or 'output'
            first = response.candidates[0]
            answer = getattr(first, "content", getattr(first, "output", str(first))).strip()
        else:
            # fallback str()
            answer = str(response).strip()
    except Exception as e:
        answer = f"Error calling Gemini: {str(e)}"

    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user,
        "question": question,
        "answer": answer,
        "sources": sources
    })
