import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()
# === CONFIG ===
TEXT_DIR = "./extracted_texts"
COLLECTION_NAME = "vdpo_documents"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# === STEP 1: Load documents ===
documents = []
for filename in os.listdir(TEXT_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(TEXT_DIR, filename), "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({"text": content, "source": filename})

print(f"📄 Loaded {len(documents)} documents.")

# === STEP 2: Chunk the text ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunks = []
for doc in documents:
    for chunk in splitter.split_text(doc["text"]):
        if len(chunk.strip()) >= 30:  # skip short chunks
            chunks.append({"text": chunk.strip(), "metadata": {"source": doc["source"]}})

print(f"🧩 Created {len(chunks)} chunks.")

# === STEP 3: Generate Embeddings ===
print("🔍 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dim vectors

texts = [chunk["text"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]

vectors = []
BATCH_SIZE = 128
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="📦 Embedding in batches"):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_vectors = model.encode(batch_texts, show_progress_bar=False).tolist()
    vectors.extend(batch_vectors)

payloads = []
for idx in range(len(texts)):
    payloads.append({
        "text": texts[idx],
        "source": metadatas[idx]["source"]
    })

# === STEP 4: Upload to Qdrant ===
print("🗃️ Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

print("🚀 Uploading vectors to Qdrant...")
client.upload_collection(
    collection_name=COLLECTION_NAME,
    vectors=vectors,
    payload=payloads,
    ids=None,
    batch_size=64
)

print("✅ All vectors embedded and uploaded to Qdrant Cloud.")
