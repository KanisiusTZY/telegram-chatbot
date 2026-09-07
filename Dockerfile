FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (ffmpeg is essential for yt-dlp audio/video extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8099

CMD ["python", "main.py"]

