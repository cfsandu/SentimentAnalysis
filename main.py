import tensorflow as tf
import streamlit as st
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

st.title('Sentiment Analyser App')
st.write('Welcome to my sentiment analysis app!')

form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')

data = ["I love you", "I hate you","We are very hayy to show you that"]

model_name = st.sidebar.selectbox("Select Model",("distilbert-base-uncased-finetuned-sst-2-english", "finiteautomata/bertweet-base-sentiment-analysis"))
#model_name ="distilbert-base-uncased-finetuned-sst-2-english"
#model_name = "finiteautomata/bertweet-base-sentiment-analysis"

#####-------IN LUCRU---------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

#token_ids = tokenizer(data, padding=True, return_tensors='tf')
#out = model(token_ids)

#Y_probas = tf.keras.activations.softmax(out.logits)
#Y_pred = tf.argmax(Y_probas, axis=1)
#print(Y_pred)

####-----------------------------------------------------------
def parse_input(ui):
    SPLIT = ','

    lst = list(ui.split(SPLIT))
    yield lst

dfdict = {}
txtlst = []
labellst = []
scorelst = []
if submit:
    model = pipeline(model=model_name)
    #lst = list(user_input.split(","))
    #for sentence in lst:
    it = parse_input(user_input) #...NICER
    for sentence in next(it):
        #res = model(sentence)
        res = clf(sentence) #...NICER
        txtlst.append(sentence)
        st.write(sentence)

        label = res[0]['label']
        labellst.append(label)
        st.write(f'label is {label}')

        score = res[0]['score']
        scorelst.append(score)
        st.write(f'score = {score}')

    dfdict['TEXT'] = txtlst
    dfdict['LABEL'] = labellst
    dfdict['SCORE'] = scorelst

    outdf = pd.DataFrame.from_dict(dfdict)
    st.write(outdf)
    
    #result = model(user_input)[0]
    #res = model(user_input)
    #st.write(result)
    #label = result['label']
    #score = result['score']

