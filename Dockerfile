FROM python:3.9-slim

WORKDIR /workspace

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME MlOpsLab

CMD ["python", "mlops_exam.py"]

