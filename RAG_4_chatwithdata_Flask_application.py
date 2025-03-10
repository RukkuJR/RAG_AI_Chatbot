import os
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ollama

# Define folder where PDFs are stored
pdf_folder = r"C:\Users\LENOVO\Downloads\Danone_Annual Results"

# Function to extract text from PDFs
def extract_text_from_pdfs(folder_path):
    all_text = []
    
    for filename in sorted(os.listdir(folder_path)):  # Ensure order (2021, 2022, 2023)
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {pdf_path}")
            
            pdf_reader = pypdf.PdfReader(pdf_path)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            all_text.append({"filename": filename, "text": text})
    
    return all_text

# Extract text from PDFs
documents = extract_text_from_pdfs(pdf_folder)

# Split text into chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in documents:
    splits = text_splitter.split_text(doc["text"])
    for chunk in splits:
        chunks.append({"text": chunk, "source": doc["filename"]})

# Delete and reinitialize the database
shutil.rmtree("./chroma_db", ignore_errors=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="Receipts_1")

# Step 2: Fetch existing IDs and delete them
existing_items = collection.get()  # Retrieve all stored items
if "ids" in existing_items and existing_items["ids"]:
    collection.delete(ids=existing_items["ids"])  # Delete using IDs
    print(f"Deleted {len(existing_items['ids'])} existing records.")

# Load Sentence Transformer model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store chunks in ChromaDB
for idx, chunk in enumerate(chunks):
    collection.add(
        ids=[str(idx)],
        documents=[chunk["text"]],
        metadatas=[{"source": chunk["source"]}]
    )

print("✅ Documents stored in ChromaDB!")


# Flask App
app = Flask(__name__, template_folder="flask")
CORS(app)  # Enable CORS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global collection  # Ensure collection is accessible
    
    data = request.json
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    # Generate ChromaDB search query
    query_response = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": f"Generate a ChromaDB search query for: {user_question}"}
    ])
    
    chroma_query = query_response.get("message", {}).get("content", "").strip()
    
    if not chroma_query:
        return jsonify({"error": "Failed to generate search query"}), 500

    # Query ChromaDB
    try:
        results = collection.query(query_texts=[chroma_query], n_results=5)
    except Exception as e:
        return jsonify({"error": f"ChromaDB query failed: {str(e)}"}), 500

    # Extract response data
    extracted_texts = []
    if "documents" in results and results["documents"]:
        extracted_texts = [doc for doc_list in results["documents"] for doc in doc_list if doc]

    context = " ".join(extracted_texts) if extracted_texts else "No relevant data found."

    # 🔥 Debugging: Print retrieved context to check ChromaDB response
    print(f"Retrieved context: {context}")

    # Ask LLaMA 3.2 for the final response
    final_response = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": f"Answer this based on the provided information: {context}.\n\nQuestion: {user_question}"}
    ])

    answer = final_response.get("message", {}).get("content", "No response generated")
    
    # 🔥 Debugging: Print final response
    print(f"Final AI Response: {answer}")

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)

