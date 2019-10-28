FROM python:3.7

COPY tests/requirements.txt requirements-test.txt
COPY requirements.txt requirements.txt

RUN pip install -r requirements-test.txt &&\ 
    pip install -r requirements.txt
