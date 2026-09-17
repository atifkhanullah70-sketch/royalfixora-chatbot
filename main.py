import os
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables (your Groq API key)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Royal Fixora AI Receptionist")

# Allow frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model (runs locally, free)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Initialize ChromaDB (local vector database)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="royalfixora")

# Base directory of this file
BASE_DIR = Path(__file__).parent


def load_pdfs_from_folder(folder_path: str):
    """Read all PDFs in a folder and return list of (text, source) chunks."""
    chunks = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(folder_path, filename)
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # Split into chunks of ~500 characters with 50 char overlap
        chunk_size = 500
        overlap = 50
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": filename
                })
            start = end - overlap
    return chunks


def build_knowledge_base():
    """Load PDFs, embed them, and store in ChromaDB."""
    print("Loading PDFs from knowledge folder...")
    chunks = load_pdfs_from_folder("knowledge")
    print(f"Found {len(chunks)} chunks.")

    if len(chunks) == 0:
        print("No PDFs found in knowledge folder.")
        return

    # Check if already indexed
    existing = collection.count()
    if existing > 0:
        print(f"Knowledge base already has {existing} chunks. Skipping rebuild.")
        return

    # Embed and store
    print("Embedding chunks (this may take a minute)...")
    texts = [c["text"] for c in chunks]
    sources = [c["source"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": s} for s in sources],
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB.")


@app.on_event("startup")
def startup_event():
    """Run knowledge base build when server starts."""
    build_knowledge_base()


@app.get("/")
def root():
    return {"status": "Royal Fixora AI Receptionist is running."}


@app.post("/chat")
def chat(question: str):
    """Answer a question using the knowledge base."""
    question = question.strip()

    if not question:
        return {
            "answer": "Please type a question and I'll do my best to help.",
            "sources": []
        }

    # Handle simple greetings without hitting the vector DB
    greetings = ["hi", "hello", "hey", "salam", "assalam", "assalamualaikum",
                 "good morning", "good evening", "good afternoon", "thanks",
                 "thank you", "shukriya", "ok", "okay"]
    if question.lower().strip("!?.,") in greetings:
        return {
            "answer": "Hello! Welcome to Royal Fixora. I can help you with:\n\n"
                      "• Service prices (plumbing, electrical, cleaning, painting)\n"
                      "• Service areas in Islamabad & Rawalpindi\n"
                      "• Booking and payment questions\n\n"
                      "What would you like to know?",
            "sources": []
        }

    # Embed the question
    question_embedding = embedder.encode([question]).tolist()

    # Search ChromaDB for top 3 relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    if not results["documents"] or not results["documents"][0]:
        return {
            "answer": "I'm not sure about that. Please contact us on WhatsApp at 0300-1234567 and our team will help you.",
            "sources": []
        }

    # Build context
    context_parts = []
    sources = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(doc)
        if meta["source"] not in sources:
            sources.append(meta["source"])

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a friendly AI receptionist for Royal Fixora, a home services company in Islamabad and Rawalpindi.

RULES:
1. Answer using ONLY the information provided below.
2. If the answer is not in the information, politely say you don't have that detail and give the WhatsApp number 0300-1234567.
3. Be warm, brief, and professional. Use 1-3 short sentences.
4. Always give exact PKR prices when asked about cost.
5. If the user asks to book or schedule, tell them to message us on WhatsApp at 0300-1234567.
6. Never invent information that isn't in the context.

INFORMATION:
{context}

CUSTOMER QUESTION: {question}

ANSWER:"""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        answer = "I'm having a small technical issue. Please try again in a moment, or contact us on WhatsApp at 0300-1234567."

    return {
        "answer": answer,
        "sources": sources
    }


@app.get("/chat.html")
def serve_chat():
    return FileResponse(BASE_DIR / "chat.html")