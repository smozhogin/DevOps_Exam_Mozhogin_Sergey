FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app/ .
COPY models/ ./models/
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]