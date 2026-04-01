FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2-binary, lxml, and document processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure download directory exists
RUN mkdir -p /app/download

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
