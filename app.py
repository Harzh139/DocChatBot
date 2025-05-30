import os
import sqlite3
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import docx
import PyPDF2
from io import BytesIO
import textwrap
from flask import Flask, request, jsonify, render_template, send_file, make_response, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from functools import wraps

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecret')
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max
    'DATABASE': 'users.db',
    'SESSION_COOKIE_SECURE': True,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'PERMANENT_SESSION_LIFETIME': timedelta(days=1),
    'ALLOWED_EXTENSIONS': {'pdf', 'docx', 'txt', 'csv', 'xlsx'},
    'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
    'MODEL': 'llama3-70b-8192',
    'GROQ_API_URL': 'https://api.groq.com/openai/v1/chat/completions'
})
print("GROQ_API_KEY loaded:", repr(app.config['GROQ_API_KEY']))
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

def get_db():
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                extracted_text TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        db.commit()
init_db()

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    db.close()
    if user:
        return User(id=user['id'], username=user['username'], email=user['email'])
    return None

def login_required_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        try:
            # Try extracting text with PyPDF2 first
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = '\n'.join([page.extract_text() or '' for page in reader.pages])
            # If no text found, try OCR
            if not text.strip():
                try:
                    from pdf2image import convert_from_path
                    import pytesseract
                    images = convert_from_path(filepath)
                    ocr_text = []
                    for img in images:
                        ocr_text.append(pytesseract.image_to_string(img))
                    text = '\n'.join(ocr_text)
                except Exception as ocr_err:
                    text = f"OCR failed: {ocr_err}"
            return text
        except Exception as e:
            return f"PDF extraction failed: {e}"
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif filepath.endswith('.csv') or filepath.endswith('.xlsx'):
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            analysis = []
            analysis.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            analysis.append(f"Columns: {', '.join(df.columns)}")
            analysis.append("\nColumn Types:")
            analysis.append(str(df.dtypes))
            analysis.append("\nMissing Values per Column:")
            analysis.append(str(df.isnull().sum()))
            outlier_report = []
            for col in df.select_dtypes(include=[np.number]).columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower) | (df[col] > upper)][col]
                outlier_report.append(f"{col}: {len(outliers)} outliers")
            if outlier_report:
                analysis.append("\nOutlier Report:")
                analysis.extend(outlier_report)
            return '\n'.join(analysis)
        except Exception as e:
            return f"Error reading data file: {str(e)}"
    return ""

def call_groq_with_messages(messages, max_tokens=1024):
    api_key = app.config['GROQ_API_KEY']
    model = app.config['MODEL']
    url = app.config['GROQ_API_URL']
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "top_p": 1
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json:
            return response_json['choices'][0]['message']['content']
        elif "error" in response_json:
            raise Exception(f"Groq API error: {response_json['error'].get('message', 'Unknown error')}")
        else:
            raise Exception(f"Unexpected response: {response_json}")
    except Exception as e:
        print("Groq API error:", str(e))
        raise

def call_groq(prompt, system_message=None, max_tokens=1024):
    return call_groq_with_messages(
        messages=[
            {"role": "system", "content": system_message or "You're a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
        else:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
        if not username or not email or not password:
            if request.is_json:
                return jsonify({"error": "All fields are required"}), 400
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
        db = get_db()
        try:
            password_hash = generate_password_hash(password)
            db.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            db.commit()
            if request.is_json:
                return jsonify({"message": "Registration successful! Please log in."})
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            if request.is_json:
                return jsonify({"error": "Username or email already exists"}), 400
            flash('Username or email already exists', 'error')
        finally:
            db.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            identifier = data.get('username') or data.get('email')
            password = data.get('password')
        else:
            identifier = request.form.get('username') or request.form.get('email')
            password = request.form.get('password')
        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?', (identifier, identifier)
        ).fetchone()
        db.close()
        if user and check_password_hash(user['password_hash'], password):
            user_obj = User(id=user['id'], username=user['username'], email=user['email'])
            login_user(user_obj)
            if request.is_json:
                return jsonify({"message": "Login successful", "username": user['username']})
            else:
                next_page = request.args.get('next')
                return redirect(next_page or url_for('home'))
        else:
            if request.is_json:
                return jsonify({"error": "Invalid username/email or password"}), 401
            flash('Invalid username/email or password', 'error')
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
@login_required_json
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@app.route('/api/check_auth')
def check_auth():
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "username": current_user.username
        })
    return jsonify({"authenticated": False})

@app.route('/upload', methods=['POST'])
@login_required_json
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
            os.makedirs(user_upload_dir, exist_ok=True)
            filepath = os.path.join(user_upload_dir, filename)
            file.save(filepath)
            extracted_text = extract_text_from_file(filepath)
            db = get_db()
            db.execute(
                'INSERT INTO documents (user_id, filename, filepath, extracted_text) VALUES (?, ?, ?, ?)',
                (current_user.id, filename, filepath, extracted_text)
            )
            db.commit()
            db.close()
            return jsonify({
                "message": "File uploaded and text extracted successfully",
                "filename": filename,
                "text_preview": textwrap.shorten(extracted_text, width=300, placeholder="...")
            })
        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/upload_link', methods=['POST'])
@login_required_json
def upload_link():
    print("Upload link endpoint called")
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": f"Failed to fetch URL (status {resp.status_code})"}), 400
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = ' '.join(soup.stripped_strings)
        # Extract the page title
        page_title = soup.title.string.strip() if soup.title and soup.title.string else url
        if not text or len(text) < 100:
            return jsonify({"error": "Could not extract meaningful content from the link."}), 400
        db = get_db()
        db.execute(
            'INSERT INTO documents (user_id, filename, filepath, extracted_text) VALUES (?, ?, ?, ?)',
            (current_user.id, page_title, url, text)
        )
        db.commit()
        db.close()
        return jsonify({
            "message": "Link content fetched and analyzed successfully.",
            "filename": page_title,
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "page_title": page_title
        })
    except Exception as e:
        return jsonify({"error": f"Error processing link: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
@login_required_json
def ask_question():
    try:
        data = request.json
        question = data.get('question')
        use_context = data.get('use_context', True)
        if not question:
            return jsonify({"error": "No question provided"}), 400
        db = get_db()
        document = db.execute(
            'SELECT extracted_text FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 1',
            (current_user.id,)
        ).fetchone()
        if not document or not document['extracted_text'].strip():
            return jsonify({"error": "No document uploaded or text extracted yet"}), 400

        # Detect if user is asking for suggestions
        suggestion_keywords = ['suggestion', 'suggest', 'recommend', 'advice', 'how to', 'ideas']
        is_suggestion = any(kw in question.lower() for kw in suggestion_keywords)

        if use_context:
            previous_questions = db.execute(
                '''SELECT question, answer FROM chat_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC LIMIT 3''',
                (current_user.id,)
            ).fetchall()
            previous_questions = reversed(previous_questions)
        else:
            previous_questions = []

        # Build the prompt
        if is_suggestion:
            system_prompt = (
                "You are an expert assistant. Based on the following document content, "
                "provide actionable suggestions, recommendations, or ideas to help the user accomplish their task. "
                "Be specific and practical.\n\n"
                f"Document Content:\n{document['extracted_text']}\n"
            )
        else:
            system_prompt = (
                "You're an expert document analyst. Answer questions based strictly on the provided document content.\n"
                f"Current Document Content:\n{document['extracted_text']}\n"
                "If the answer is not in the document, say \"I don't know.\"."
            )

        messages = [{
            "role": "system",
            "content": system_prompt
        }]
        for qa in previous_questions:
            messages.append({
                "role": "user",
                "content": qa['question']
            })
            messages.append({
                "role": "assistant",
                "content": qa['answer']
            })
        messages.append({
            "role": "user",
            "content": question
        })

        response = call_groq_with_messages(
            messages=messages,
            max_tokens=1500
        )
        fallback_triggers = [
            "i don't know.", "i don't know", "not found in the document.",
            "not found in the document", "", None
        ]
        if response is None or response.strip().lower() in fallback_triggers:
            fallback_messages = [
                {"role": "system", "content": "You're a helpful assistant. Answer the user's question as best as you can."},
                {"role": "user", "content": question}
            ]
            fallback_response = call_groq_with_messages(
                messages=fallback_messages,
                max_tokens=1500
            )
            if (
                fallback_response is None or
                fallback_response.strip() == "" or
                fallback_response.strip().lower() in fallback_triggers
            ):
                friendly_msg = "Sorry, I couldn't understand your question. Could you please rephrase it or ask more clearly?"
                db.execute(
                    'INSERT INTO chat_history (user_id, question, answer) VALUES (?, ?, ?)',
                    (current_user.id, question, friendly_msg)
                )
                db.commit()
                db.close()
                return jsonify({
                    "answer": friendly_msg,
                    "model": app.config['MODEL'],
                    "context_used": False,
                    "source": "fallback"
                })
            db.execute(
                'INSERT INTO chat_history (user_id, question, answer) VALUES (?, ?, ?)',
                (current_user.id, question, fallback_response)
            )
            db.commit()
            db.close()
            return jsonify({
                "answer": fallback_response,
                "model": app.config['MODEL'],
                "context_used": False,
                "source": "external"
            })
        db.execute(
            'INSERT INTO chat_history (user_id, question, answer) VALUES (?, ?, ?)',
            (current_user.id, question, response)
        )
        db.commit()
        db.close()
        return jsonify({
            "answer": response,
            "model": app.config['MODEL'],
            "context_used": use_context,
            "source": "document"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "solution": "Please try again later"
        }), 500

@app.route('/download/text', methods=['GET'])
@login_required
def download_text():
    format = request.args.get('format', 'txt').lower()
    db = get_db()
    document = db.execute(
        'SELECT extracted_text FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 1',
        (current_user.id,)
    ).fetchone()
    db.close()
    if not document or not document['extracted_text'].strip():
        flash('No document text available to download', 'error')
        return redirect(url_for('home'))
    try:
        if format == 'txt':
            response = make_response(document['extracted_text'])
            response.headers['Content-Type'] = 'text/plain'
            response.headers['Content-Disposition'] = f'attachment; filename=extracted_text_{current_user.id}.txt'
            return response
        elif format == 'pdf':
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            text = p.beginText(40, 750)
            text.setFont("Helvetica", 12)
            for line in document['extracted_text'].split('\n'):
                for wrapped_line in textwrap.wrap(line, width=100):
                    text.textLine(wrapped_line)
            p.drawText(text)
            p.showPage()
            p.save()
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'extracted_text_{current_user.id}.pdf',
                mimetype='application/pdf'
            )
        else:
            flash('Invalid format. Use "txt" or "pdf"', 'error')
            return redirect(url_for('home'))
    except Exception as e:
        flash(f'Failed to generate download: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/download/chat', methods=['GET'])
@login_required
def download_chat():
    format = request.args.get('format', 'txt').lower()
    db = get_db()
    chats = db.execute(
        'SELECT question, answer FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC',
        (current_user.id,)
    ).fetchall()
    db.close()
    if not chats:
        flash('No chat history available to download', 'error')
        return redirect(url_for('home'))
    try:
        chat_text = f"Chat History for {current_user.username}\n\n"
        for chat in chats:
            chat_text += f"Q: {chat['question']}\n"
            chat_text += f"A: {chat['answer']}\n\n"
            chat_text += "-" * 50 + "\n\n"
        if format == 'txt':
            response = make_response(chat_text)
            response.headers['Content-Type'] = 'text/plain'
            response.headers['Content-Disposition'] = f'attachment; filename=chat_history_{current_user.id}.txt'
            return response
        elif format == 'pdf':
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            text = p.beginText(40, 750)
            text.setFont("Helvetica", 12)
            for line in chat_text.split('\n'):
                for wrapped_line in textwrap.wrap(line, width=100):
                    text.textLine(wrapped_line)
            p.drawText(text)
            p.showPage()
            p.save()
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'chat_history_{current_user.id}.pdf',
                mimetype='application/pdf'
            )
        else:
            flash('Invalid format. Use "txt" or "pdf"', 'error')
            return redirect(url_for('home'))
    except Exception as e:
        flash(f'Failed to generate download: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/chat/history', methods=['GET'])
@login_required_json
def get_chat_history():
    try:
        limit = request.args.get('limit', default=10, type=int)
        offset = request.args.get('offset', default=0, type=int)
        db = get_db()
        chats = db.execute(
            '''SELECT id, question, answer, timestamp 
            FROM chat_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?''',
            (current_user.id, limit, offset)
        ).fetchall()
        total_chats = db.execute(
            'SELECT COUNT(*) FROM chat_history WHERE user_id = ?',
            (current_user.id,)
        ).fetchone()[0]
        db.close()
        return jsonify({
            "chats": [dict(chat) for chat in chats],
            "total": total_chats,
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat/clear', methods=['POST'])
@login_required_json
def clear_chat_history():
    try:
        db = get_db()
        db.execute(
            'DELETE FROM chat_history WHERE user_id = ?',
            (current_user.id,)
        )
        db.commit()
        db.close()
        return jsonify({"message": "Chat history cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/profile')
@login_required
def profile():
    db = get_db()
    doc_count = db.execute(
        'SELECT COUNT(*) FROM documents WHERE user_id = ?',
        (current_user.id,)
    ).fetchone()[0]
    chat_count = db.execute(
        'SELECT COUNT(*) FROM chat_history WHERE user_id = ?',
        (current_user.id,)
    ).fetchone()[0]
    recent_docs = db.execute(
        'SELECT filename, uploaded_at FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5',
        (current_user.id,)
    ).fetchall()
    db.close()
    return jsonify({
        "username": current_user.username,
        "email": current_user.email,
        "doc_count": doc_count,
        "chat_count": chat_count,
        "recent_docs": [dict(doc) for doc in recent_docs]
    })

@app.route('/templates/<template_name>')
def serve_template(template_name):
    if template_name not in ['index.html', 'login.html', 'register.html', 'profile.html']:
        return "Template not found", 404
    return render_template(template_name)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)