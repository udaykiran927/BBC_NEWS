from flask import Flask,render_template,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

app=Flask(__name__)
df=pd.read_csv("BBC_NEWS.csv")
data=pd.read_csv("data.csv")
model=pickle.load(open("model.pkl","rb"))
stopwords = nltk.corpus.stopwords.words('english')
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
features = tfidf.fit_transform(data.news_porter_stemmed).toarray()

@app.route("/")

def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST","GET"])

def predict():
    inputs=[i for i in request.form.values()]
    id_to_category = {0:"business",1:"tech",2:"politics",3:"sport",4:"entertainment"}
    test_article =inputs[0]
    print(test_article)
    test_article = test_article.lower()
    test_article=' '.join([word for word in test_article.split() if word not in (stopwords)])
    k=test_article.split()
    test_frame = pd.DataFrame({"Text":[test_article]})
    test_feature = tfidf.transform(test_frame.Text).toarray()
    print(test_feature)
    prediction = model.predict(test_feature)
    predicted=id_to_category[prediction[0]]
    return render_template("index.html",pred="The Article is About : {}".format(predicted))



if __name__=='__main__':
    app.run(debug=True)