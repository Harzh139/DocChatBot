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

# ------------------------ Utility Functions ------------------------

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
        chunk_size=1000,      # larger chunk for better coherence
        chunk_overlap=200,    # overlap helps LLM keep context
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

def get_relevant_context(question, top_k=5):
    question_lower = question.lower()
    question_embedding = embedder.embed_query(question)
    D, I = index.search(np.array([question_embedding]).astype('float32'), k=top_k)
    matched_chunks = [document_chunks[i] for i in I[0]]

    # Step 1: Keyword Boosting
    keywords = set(question_lower.split())
    priority_chunks = []

    # Boost chunks that contain keywords, accuracy info, or known model names
    for chunk in matched_chunks:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in keywords) or \
           "inception-resnet" in chunk_lower or \
           "%" in chunk_lower or \
           "accuracy" in chunk_lower:
            priority_chunks.append(chunk)

    # Step 2: Return top filtered or fallback to matched
    selected_chunks = priority_chunks if priority_chunks else matched_chunks

    # Deduplicate chunks
    seen = set()
    unique_chunks = []
    for chunk in selected_chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)

    return "\n---\n".join(unique_chunks)


def generate_rag_response(question, context):
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.
If the question is about a section like 'Introduction' or 'Conclusion', find and summarize that section from the context if possible.


Context:
{context}

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

# ------------------------ Flask Routes ------------------------

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
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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
        context = get_relevant_context(question)
        answer = generate_rag_response(question, context)
        return jsonify({
            'question': question,
            'answer': answer,
            'context_used': context
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/summary', methods=['GET'])
def summarize():
    if not document_chunks:
        return jsonify({'error': 'No document uploaded'}), 400

    try:
        context = "\n".join(document_chunks[:5])  # Take first few chunks
        prompt = f"Summarize the following research paper content:\n{context}"
        summary = llm(prompt, max_length=250, do_sample=False, temperature=0.3)[0]['generated_text']
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ------------------------ Run App ------------------------

if __name__ == '__main__':
    print("Starting app...")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=True, host='0.0.0.0', port=port)
