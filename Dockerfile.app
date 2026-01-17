FROM python:3.11-slim
WORKDIR /app
COPY requirements_app.txt .
RUN python -m pip install --no-cache-dir -r requirements_app.txt
COPY app .
COPY params.yaml .
COPY src ./src
COPY models ./models
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]