# syntax=docker/dockerfile:1
# Builds an image with the Python 3.12 image
FROM python:3.12-alpine
# Sets the working directory to `/code`
WORKDIR /code
ENV FLASK_APP=app/main.py
ENV FLASK_RUN_HOST=0.0.0.0
# Installs gcc and other dependencies
RUN apk add --no-cache gcc musl-dev linux-headers
# Copies requirements.txt and installs Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
# Copies the current directory into the workdir
COPY . .
EXPOSE 5000
CMD ["flask", "run", "--debug"]  # Sets the default command for the container to `flask run --debug`