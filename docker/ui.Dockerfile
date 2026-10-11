# Streamlit UI image — tiny: it only needs streamlit + requests (thin client).
#   docker build -f docker/ui.Dockerfile -t boot-search-ui .
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PORT=8501 API_URL=http://api:8000
RUN useradd --create-home --uid 10001 app
WORKDIR /srv
COPY ui/requirements.txt .
RUN pip install -r requirements.txt
COPY ui/ ui/
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"
CMD ["sh", "-c", "streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true --browser.gatherUsageStats false"]
