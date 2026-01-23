FROM python:3.12-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    pandas>=2.0.0 \
    tqdm>=4.66.0


COPY . .


RUN mkdir -p models data


EXPOSE 8000


ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]