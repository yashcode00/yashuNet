FROM alpine:latest

RUN apk add build-base
RUN apk add --update py3-pip
RUN apk add python3-dev py3-numpy py3-pandas
WORKDIR /app/

RUN mkdir -p instance/
COPY static/* ./static/
COPY templates/* ./templates/
COPY *.py ./
COPY requirements.txt ./

RUN python3 -m venv venv
RUN source venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

ENV SECRET_KEY = '_5#y2L"F4Q8z\n\xec]/'
ENV DATABASE_PATH = 'instance/database.db'

EXPOSE 5000
CMD ["venv/bin/python", "app.py"]