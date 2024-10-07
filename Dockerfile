FROM tensorflow/tensorflow:2.16.1

RUN pip install matplotlib python-dotenv

WORKDIR /app