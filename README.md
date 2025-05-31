# Document Chatbot

A secure, user-friendly chatbot for document Q&A and data analysis. Supports PDF, DOCX, TXT, CSV, XLSX. Uses Groq LLM (Llama 3) for fallback answers.

---

## 1. Models & Data

**Database Tables**
- **users**: Stores user credentials (username, email, password hash, created_at).
- **documents**: Stores uploaded documents (user_id, filename, extracted text, upload time).
- **chat_history**: Stores Q&A pairs for each user (user_id, question, answer, timestamp).

**Data Processing**
- **Text Extraction**: Supports PDF, DOCX, TXT, CSV, XLSX.
- **CSV/XLSX**: Uses pandas (and openpyxl for .xlsx) to analyze data (shape, columns, types, missing values, outliers).
- **Document Analysis**: Extracted text or data summary is stored and used for Q&A.

---

## 2. APIs & Endpoints

**Authentication**
- `/register`: Register a new user (JSON or form data).
- `/login`: Log in a user (JSON or form data).
- `/logout`: Log out the current user.
- `/api/check_auth`: Check if the user is authenticated (for frontend session management).

**Document Upload & Processing**
- `/upload`: Upload a document (PDF, DOCX, TXT, CSV, XLSX), extract text or data summary, and store it.
  - **Image-based PDFs are not supported**; users are notified if they try to upload one.

**Chat & Q&A**
- `/ask`: Main endpoint for asking questions about the uploaded document.
  - First, tries to answer using the document content (contextual Q&A, with chunking for large files).
  - If not found, falls back to the Groq API for a general answer and notifies the user.
  - For suggestion/advice questions, uses a larger context window for actionable recommendations.
- `/chat/history`: Returns the user's chat history.
- `/chat/clear`: Clears the user's chat history.

**Download**
- `/download/text`: Download the extracted document text as TXT or PDF.
- `/download/chat`: Download the chat history as TXT or PDF.

**Profile**
- `/profile`: Returns user profile info, document count, chat count, and recent documents.

---

## 3. External API: Groq

- **Groq API**: [https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)
- **Model**: llama3-70b-8192 (configurable)
- Used for LLM-powered Q&A and fallback answers.
- Called via `call_groq_with_messages()`.

---

## 4. Main Methods

- **extract_text_from_file(filepath)**: Handles extraction and analysis for all supported file types. Rejects image-based PDFs with a user-friendly message.
- **call_groq_with_messages(messages, max_tokens)**: Sends a chat completion request to the Groq API.
- **ask_question()**: Handles Q&A logic, including context, chunking, fallback, and user-friendly error messages.
- **allowed_file(filename)**: Checks if the uploaded file is of an allowed type.

---

## 5. Frontend

- Single-page app with login/register, upload, chat, and download features.
- File input supports PDF, DOCX, TXT, CSV, XLSX.
- Download buttons for extracted text and chat history.
- User info and logout always visible.
- Handles and displays user-friendly error messages.

---

## 6. Key Improvements

- **Robust large file handling**: Chunking logic for efficient processing of big documents.
- **Smart fallback to external API**: If info is not in the doc, the bot calls Groq LLM and notifies the user.
- **Improved CSV/XLSX support**: Data analysis, missing values, outlier detection, and openpyxl support.
- **Clear handling of image-based PDFs**: OCR is disabled; users are informed if they upload a scanned PDF.
- **Actionable suggestions**: For advice/recommendation questions, uses more context for better answers.
- **Consistent error handling**: All API endpoints return JSON errors for frontend compatibility.
- **Production-ready deployment**: Updated Dockerfile and requirements for reliable deployment.

---

## 7. Tech Stack

- **Backend**: Flask, Flask-Login, SQLite, pandas, openpyxl, PyPDF2, python-docx, pytesseract (OCR disabled for image PDFs)
- **Frontend**: HTML, CSS, JavaScript (single-page app)
- **LLM**: Groq API (Llama 3)
- **Deployment**: Docker, Render (or any Docker-compatible platform)

---

## 8. Usage Notes

- Only text-based PDFs are supported (image-based/scanned PDFs are rejected).
- For large files, the bot processes in chunks for efficiency.
- If the answer is not in the document, the bot will fetch info from the LLM and clearly inform the user.

---