import os
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# === CONFIGURATION ===
PDF_FOLDER = "papers/"
INDEX_FOLDER = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
OCR_FALLBACK = True

# === Initialize text splitter and embedding model ===
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# === Extract Text from PDF (per page) ===
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

# === Process PDFs and Store in FAISS ===
def process_pdfs_and_store():
    documents = []
    metadatas = []

    for filename in os.listdir(PDF_FOLDER):
        if not filename.endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"Processing {filename}...")

        page_texts = extract_text_per_page(pdf_path)
        if not page_texts:
            print(f"⚠️ Skipping empty PDF: {filename}")
            continue

        for page_number, page_text in page_texts:
            if not page_text.strip():
                continue
            chunks = text_splitter.split_text(page_text)
            for chunk_id, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_id": chunk_id
                })

    # Create FAISS DB
    vectorstore = FAISS.from_texts(texts=documents, embedding=embedding_model, metadatas=metadatas)
    vectorstore.save_local(INDEX_FOLDER)
    print("✅ All PDFs processed and saved to FAISS.")

# === Hybrid Search (semantic + keyword) ===
def hybrid_search(query, top_k=5):
    vectorstore = FAISS.load_local(INDEX_FOLDER, embedding_model, allow_dangerous_deserialization=True)
    
    # Vector Search
    vector_hits = vectorstore.similarity_search(query, k=top_k * 2)

    # Keyword scoring based on content
    scored = []
    for doc in vector_hits:
        keyword_score = doc.page_content.lower().count(query.lower())
        scored.append((keyword_score, doc))

    # Sort by keyword relevance
    scored = sorted(scored, key=lambda x: -x[0])

    # Return top_k documents
    return [doc for _, doc in scored[:top_k]]

# === Entry Point ===
if __name__ == "__main__":
    # Perform extraction and indexing once at startup
    process_pdfs_and_store()

    # Example user query
    query = str(input("Enter your search query: ").strip())
    k = int(input("Enter number of top results to return (default 3): ") or 3)
    top_k = k if k > 0 else 3
    print(f"\n🔍 Searching for: '{query}' with top {top_k} results...\n"
    )
    if not query:
        print("❗ No query provided. Exiting.")
        exit()

    results = hybrid_search(query, top_k=top_k)

    print("\n🔍 Top Results:\n")
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"Result {i+1}")
        print(f"File: {meta['filename']} | Page: {meta['page_number']} | Chunk: {meta['chunk_id']}")
        print(doc.page_content[:400], "\n---\n")
