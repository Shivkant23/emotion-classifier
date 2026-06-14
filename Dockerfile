FROM python:3.11-slim

ARG HF_MODEL_NAME=Shubham231/distilbert_v1

ENV HF_MODEL_NAME=${HF_MODEL_NAME}
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .

CMD ["python", "predict.py"]