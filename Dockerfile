# syntax=docker/dockerfile:1
# Use slim Debian-based image — Alpine lacks pre-built wheels for mlflow/scipy/xgboost
FROM python:3.12-slim
# Sets the working directory to `/code`
WORKDIR /code
ENV FLASK_APP=app/main.py
ENV FLASK_RUN_HOST=0.0.0.0
# Install build tools needed for compiled Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# Copies requirements.txt and installs Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copies the current directory into the workdir
COPY . .
EXPOSE 5000
CMD ["flask", "run", "--debug"]