FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . ./

RUN pip install --no-cache-dir .

CMD ["python3", "proxycache.py"]
