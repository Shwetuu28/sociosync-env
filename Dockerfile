FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn[standard] openenv-core openai pydantic>=2.0

EXPOSE 7860

CMD ["python", "app.py"]