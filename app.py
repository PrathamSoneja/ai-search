import os
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

# === CONFIGURATION ===
PDF_FOLDER = "papers/"
INDEX_FOLDER = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
OCR_FALLBACK = True

# === Initialize app, splitter, embedder ===
app = FastAPI(title="Research Paper Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore: FAISS = None  # Will be initialized at startup

# === Text Extraction per Page ===
def extract_text_per_page(pdf_path):
    page_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                page_texts.append((i + 1, page_text))
            elif OCR_FALLBACK:
                images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                ocr_text = pytesseract.image_to_string(images[0])
                page_texts.append((i + 1, ocr_text))
    return page_texts

# === Process single PDF and return texts + metadatas ===
def process_pdf(pdf_path, filename):
    documents, metadatas = [], []
    page_texts = extract_text_per_page(pdf_path)

    for page_number, page_text in page_texts:
        chunks = text_splitter.split_text(page_text)
        for chunk_id, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "page_number": page_number,
                "chunk_id": chunk_id
            })

    return documents, metadatas

# === Process all PDFs from folder ===
def initialize_faiss_from_folder():
    documents, metadatas = [], []
    for filename in os.listdir(PDF_FOLDER):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(PDF_FOLDER, filename)
        print(f"Indexing: {filename}")
        doc_chunks, meta_chunks = process_pdf(filepath, filename)
        documents.extend(doc_chunks)
        metadatas.extend(meta_chunks)

    if documents:
        faiss_store = FAISS.from_texts(texts=documents, embedding=embedding_model, metadatas=metadatas)
        faiss_store.save_local(INDEX_FOLDER)
        return faiss_store
    return None

# === Hybrid Search ===
def hybrid_search(query: str, top_k: int = 5) -> List[dict]:
    global vectorstore
    vector_hits = vectorstore.similarity_search(query, k=top_k * 2)

    scored = []
    for doc in vector_hits:
        score = doc.page_content.lower().count(query.lower())
        scored.append((score, doc))

    scored = sorted(scored, key=lambda x: -x[0])
    results = []
    for score, doc in scored[:top_k]:
        results.append({
            "filename": doc.metadata["filename"],
            "page_number": doc.metadata["page_number"],
            "chunk_id": doc.metadata["chunk_id"],
            "content": doc.page_content[:500]  # Preview
        })
    return results

# === API ENDPOINTS ===

@app.on_event("startup")
def on_startup():
    global vectorstore
    print("📥 Initializing FAISS from PDF folder...")
    os.makedirs(PDF_FOLDER, exist_ok=True)
    vectorstore = initialize_faiss_from_folder()
    if not vectorstore:
        vectorstore = FAISS.from_texts([], embedding=embedding_model)
        print("⚠️ No documents found initially.")

@app.get("/search")
def search(query: str = Query(...), top_k: int = Query(5)):
    if not query:
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

    results = hybrid_search(query, top_k=top_k)
    return {"results": results, "query": query, "top_k": top_k}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore
    if not file.filename.endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

    filepath = os.path.join(PDF_FOLDER, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    documents, metadatas = process_pdf(filepath, file.filename)
    if not documents:
        return JSONResponse({"error": "No valid content found in PDF"}, status_code=400)

    vectorstore.add_texts(texts=documents, metadatas=metadatas)
    vectorstore.save_local(INDEX_FOLDER)

    return {"message": f"File '{file.filename}' uploaded and indexed.", "chunks_added": len(documents)}
