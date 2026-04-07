FROM python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt .

COPY packages ./packages

RUN pip install --no-cache-dir --no-index --find-links=packages -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "inference.py"]