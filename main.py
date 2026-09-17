import os
from pathlib import Path
import chromadb
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pypdf import PdfReader
from groq import Groq

load_dotenv()

app = FastAPI(title="Royal Fixora AI Receptionist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")



collection = chroma_client.get_or_create_collection(name="royalfixora")

BASE_DIR = Path(__file__).parent


def get_embeddings(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        raise ValueError("JINA_API_KEY is not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jina_api_key}"
    }
    data = {
        "input": texts,
        "model": "jina-embeddings-v3",
        "task": task,
        "dimensions": 384
    }
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def load_pdfs_from_folder(folder_path: str):
    chunks = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(folder_path, filename)
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        chunk_size = 500
        overlap = 50
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "source": filename})
            start = end - overlap
    return chunks


def build_knowledge_base():
    print("Loading PDFs...")
    chunks = load_pdfs_from_folder("knowledge")
    print(f"Found {len(chunks)} chunks.")

    if not chunks:
        return

    if collection.count() > 0:
        print("Knowledge base already built.")
        return

    print("Embedding chunks via Jina AI (passage)...")
    texts = [c["text"] for c in chunks]
    sources = [c["source"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    embeddings = get_embeddings(texts, task="retrieval.passage")

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": s} for s in sources],
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks.")


@app.on_event("startup")
def startup_event():
    build_knowledge_base()


@app.get("/")
def root():
    return {"status": "Royal Fixora AI Receptionist is running."}


@app.post("/chat")
def chat(question: str):
    question = question.strip()
    if not question:
        return {"answer": "Please type a question.", "sources": []}

    greeting_words = ["hi", "hello", "hey", "salam", "assalam", "thanks",
                      "thank you", "shukriya", "ok", "okay"]
    goodbye_words = ["bye", "goodbye", "see you", "good night"]

    cleaned = question.lower().strip("!?.,")

    if cleaned in greeting_words:
        return {
            "answer": "Hello! 👋 Welcome to Royal Fixora. Ask me about our services, prices, or coverage areas.",
            "sources": []
        }

    if cleaned in goodbye_words:
        return {
            "answer": "Thank you for visiting Royal Fixora! If you need anything, message us on WhatsApp at 0300-1234567. Have a great day! 👋",
            "sources": []
        }

    question_embedding = get_embeddings([question], task="retrieval.query")[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    if not results["documents"] or not results["documents"][0]:
        return {
            "answer": "I'm not sure. Please contact us on WhatsApp at 0300-1234567.",
            "sources": []
        }

    context_parts, sources = [], []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(doc)
        if meta["source"] not in sources:
            sources.append(meta["source"])

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a friendly AI receptionist for Royal Fixora (home services in Islamabad/Rawalpindi).

Answer using ONLY the information below. If the answer is not in the information, say:
"I don't have that information. Please contact us on WhatsApp at 0300-1234567."

Be warm, brief (1-3 sentences). Give exact PKR prices when asked about cost.
Always use the WhatsApp number 0300-1234567 — never any other number.

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
        answer = "I'm having a technical issue. Please contact us on WhatsApp at 0300-1234567."

    return {"answer": answer, "sources": sources}


@app.get("/chat.html")
def serve_chat():
    possible_paths = [
        BASE_DIR / "chat.html",
        Path("chat.html"),
        Path("/app/chat.html"),
        Path.cwd() / "chat.html",
    ]
    for p in possible_paths:
        if p.exists():
            return FileResponse(p)
    return {
        "error": "chat.html not found",
        "tried_paths": [str(p) for p in possible_paths]
    }