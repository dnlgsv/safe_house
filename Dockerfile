# Use a lightweight Python base image
FROM python:3.12-slim

# Install system dependencies required for OpenCV, PyAudio, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY . .

# Define the entrypoint command
CMD ["python", "app.py"]
