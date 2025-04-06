FROM python:3.10-slim

WORKDIR /app/module_1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY module_1 /app/module_1

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
