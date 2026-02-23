FROM python:3.10-slim

WORKDIR /app

RUN pip install streamlit requests

COPY app.py .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]