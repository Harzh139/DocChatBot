FROM python:3.10-slim

WORKDIR /app

# Avoid interactive prompts
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Hugging Face Spaces requires app on port 7860
EXPOSE 7860

CMD ["python", "app.py"]
