FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ABW_Datenvisualisierung.py .

EXPOSE 8501

CMD ["streamlit", "run", "ABW_Datenvisualisierung.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
