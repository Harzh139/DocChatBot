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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize models
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = pipeline("text2text-generation", model="google/flan-t5-base")

# Global variables
index = None
document_chunks = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        text = ''
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ''
        return text
    elif ext == 'docx':
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_index(text_chunks):
    global index, document_chunks
    document_chunks = text_chunks
    embeddings = embedder.embed_documents(text_chunks)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

def generate_rag_response(question, context):
    prompt = f"""Answer this question based on the context:

Context: {context}
Question: {question}

Answer:"""
    try:
        response = llm(
            prompt,
            max_length=200,
            do_sample=False,
            temperature=0.1
        )
        return response[0]['generated_text'].strip()
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return "I encountered an error processing your question."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure uploads dir exists
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        text = extract_text_from_file(file_path)
        if not text.strip():
            return jsonify({'error': 'File is empty or could not be read'}), 400
            
        preprocessed = preprocess_text(text)
        chunks = chunk_text(preprocessed)
        create_index(chunks)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'chunk_count': len(chunks)
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    if index is None:
        return jsonify({'error': 'No document uploaded yet'}), 400
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        question_embedding = embedder.embed_query(question)
        D, I = index.search(np.array([question_embedding]).astype('float32'), k=3)
        context = "\n---\n".join([document_chunks[i] for i in I[0]])
        answer = generate_rag_response(question, context)
        return jsonify({
            'question': question,
            'answer': answer,
            'relevant_chunks': I[0].tolist(),
            'scores': D[0].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=7860)

