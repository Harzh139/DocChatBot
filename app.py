from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import faiss
import numpy as np
import re
import os
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import torch
from time import time

# ------------------------ Flask App Configuration ------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ------------------------ Initialize Optimized Models ------------------------
DEVICE = 0 if torch.cuda.is_available() else -1
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # More accurate than MiniLM

# Load models in FP16 for faster inference (if GPU available)
embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cuda' if DEVICE == 0 else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}  # Better similarity matching
)

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=DEVICE,
    torch_dtype=torch.float16 if DEVICE == 0 else torch.float32
)

# ------------------------ Global Variables ------------------------
index = None
document_chunks = []
document_embeddings = None

# ------------------------ Optimized Utility Functions ------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file_path):
    """Faster text extraction with error handling"""
    try:
        ext = file_path.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            with open(file_path, 'rb') as f:
                return ''.join([page.extract_text() or '' for page in PdfReader(f).pages])
        elif ext == 'docx':
            return '\n'.join([para.text for para in Document(file_path).paragraphs])
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
    return ""

def preprocess_text(text):
    """Optimized preprocessing"""
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    return text.strip()

def chunk_text(text):
    """Smart chunking with overlap for better context"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjusted for larger documents
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better semantic splitting
    )
    return text_splitter.split_text(text)

def create_index(text_chunks):
    global index, document_chunks, document_embeddings
    document_chunks = text_chunks
    document_embeddings = embedder.embed_documents(text_chunks)
    
    # Convert to numpy array FIRST
    document_embeddings_np = np.array(document_embeddings).astype('float32')
    
    # Then get dimension and normalize
    dimension = document_embeddings_np.shape[1]  # Now we can use .shape!
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(document_embeddings_np)  # Operate on numpy array
    
    index.add(document_embeddings_np)  # Add the normalized embeddings

def get_relevant_context(question, top_k=5):
    """Enhanced context retrieval with keyword boosting"""
    start_time = time()
    
    # Embed question
    question_embedding = embedder.embed_query(question)
    question_embedding = np.array([question_embedding]).astype('float32')
    faiss.normalize_L2(question_embedding)
    
    # Semantic search
    D, I = index.search(question_embedding, k=top_k*2)  # Get extra for filtering
    
    # Keyword boosting (case-insensitive)
    keywords = set(re.findall(r'\w+', question.lower()))
    boosted_chunks = []
    for idx in I[0]:
        chunk = document_chunks[idx]
        chunk_lower = chunk.lower()
        
        # Boost if keywords found or contains important terms
        if (any(kw in chunk_lower for kw in keywords) or \
           any(term in chunk_lower for term in ["inception-resnet", "accuracy", "%", "probability"])):
            boosted_chunks.append((idx, chunk))
    
    # Use boosted chunks if found, otherwise fallback
    selected_chunks = boosted_chunks[:top_k] if boosted_chunks else [(i, document_chunks[i]) for i in I[0][:top_k]]
    
    # Deduplicate
    seen = set()
    unique_chunks = []
    for idx, chunk in selected_chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    
    print(f"Context retrieval took {time() - start_time:.2f}s")
    return "\n---\n".join(unique_chunks[:top_k])

def format_response(text):
    """Convert raw text into structured bullet points with emojis"""
    if not text.strip():
        return "🚫 I couldn't find that information."
    
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    # Add bullet points and emojis
    emojis = ["🔹", "✅", "📌", "🌟", "📝"]  # Rotation for variety
    formatted = []
    for i, sentence in enumerate(sentences):
        emoji = emojis[i % len(emojis)]
        formatted.append(f"{emoji} {sentence}")
    
    return "\n".join(formatted)

def generate_rag_response(question, context, max_length=2000):  # Increased max_length
    """Optimized RAG response generation"""
    start_time = time()
    
    # Truncate context intelligently
    context = context[:max_length]
    
    prompt = f"""Provide a detailed answer in bullet points with emojis. Use only this context:
    
Context:
{context}

Question: {question}

Answer (5-7 bullet points max):"""  # Increased bullet points for more detail
    
    try:
        response = llm(
            prompt,
            max_new_tokens=300,  # Increased token limit for more comprehensive answers
            temperature=0.3,  # Less randomness
            do_sample=False,
            num_beams=5  # Increased beams for better quality
        )[0]['generated_text']
        
        print(f"LLM generation took {time() - start_time:.2f}s")
        return format_response(response)
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return "⚠️ Error generating response"


# ------------------------ Flask Routes ------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        text = extract_text_from_file(filepath)
        if not text.strip():
            return jsonify({'error': 'Empty file'}), 400
        
        chunks = chunk_text(preprocess_text(text))
        create_index(chunks)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'chunks': len(chunks)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    if index is None:
        return jsonify({'error': 'Upload a document first'}), 400
    
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Empty question'}), 400
    
    try:
        context = get_relevant_context(question)
        answer = generate_rag_response(question, context)
        
        return jsonify({
            'question': question,
            'answer': answer,
            'context_used': context[:500] + "..."  # Preview
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summary', methods=['GET'])
def summarize():
    if not document_chunks:
        return jsonify({'error': 'No document loaded'}), 400
    
    try:
        context = "\n".join(document_chunks[:3])  # First few chunks
        prompt = f"""Summarize key points in 5 bullet points with emojis:
        
{context}

Summary:"""
        
        summary = llm(
            prompt,
            max_new_tokens=150,
            temperature=0.2  # More deterministic
        )[0]['generated_text']
        
        return jsonify({
            'summary': format_response(summary)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=7860, debug=True)
