FROM python:3.14-slim

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]