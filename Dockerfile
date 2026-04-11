FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn openenv-core

CMD ["python", "-m", "server.app"]