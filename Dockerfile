FROM python:3.9

WORKDIR /ml/sentiment_analysis

#ADD LogisticRegr.py .
COPY . /ml/sentiment_analysis/

RUN pip install -r requirements.txt

CMD ["python", "./ml/sentiment_analysis/main.py"]