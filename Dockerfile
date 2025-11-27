# Use a lightweight Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run expects your app to listen on 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]