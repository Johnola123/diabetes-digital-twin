
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -e .

COPY configs ./configs

EXPOSE 8501

CMD ["sh", "-c", "python -m dtwin.cli run --env ${APP_ENV:-dev} && python -m streamlit run src/dtwin/dashboard.py --server.address 0.0.0.0 --server.port 8501"]
