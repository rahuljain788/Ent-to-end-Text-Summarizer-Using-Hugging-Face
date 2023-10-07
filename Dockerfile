FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN pip uninstall -y transformers accelerate
RUN pip install transformers accelerate

# CMD ["python3", "app.py"]
ENTRYPOINT ["streamlit", "run", "test_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]