FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into container
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run script
CMD ["python", "main.py"]
