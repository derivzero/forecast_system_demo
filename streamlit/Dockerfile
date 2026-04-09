FROM python:3.11-slim

WORKDIR /dev_env

COPY dev_env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dev_env/ .

EXPOSE 8501

CMD ["streamlit", "run", "app/app_pg.py", "--server.port=8501", "--server.address=0.0.0.0"]