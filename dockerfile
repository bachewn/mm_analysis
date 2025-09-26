# Base Python image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# COPY only code (no Excel)
COPY mm_core.py fastapi_app.py streamlit_app.py betting_core.py ./

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "-c", "echo 'Provide a command (uvicorn or streamlit) at runtime.'"]
