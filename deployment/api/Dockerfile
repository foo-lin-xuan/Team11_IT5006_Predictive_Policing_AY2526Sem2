# Use the exact Python version you are using locally
FROM python:3.11.7-slim

# Set environment variables to keep Python snappy
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces runs on user 1000 for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# 1. Copy requirements first (improves build caching)
COPY --chown=user deployment/api/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 2. Copy the necessary code
COPY --chown=user deployment/api/ ./

# 3. Expose the port HF expects
EXPOSE 7860

# 4. Start the app
# Note: 'main' refers to main.py inside deployment/api/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]