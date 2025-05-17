#importing libraries
import streamlit as st
import nltk
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download("punkt")
nltk.download('stopwords')

#Load model
model = pickle.load(open("model.pkl", "rb"))
#load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

#App title 
with st.sidebar:
     st.image("images.png", use_column_width =True)
     st.markdown("""
    ## 🎬 Movie Review Sentiment Analyzer  
                 
    This app predicts whether a movie review expresses a **Positive** or **Negative** sentiment using a trained Machine Learning model.
    """)
st.title("📈 Movie Sentiment Classifier")
st.write("Enter a movie review below and click **Predict** to see the sentiment.")

text = st.text_area("✍️ Your Review", height=150)

# defining function to clean input news
stop_words=set(stopwords.words("english"))
stemmer=PorterStemmer()

def data_cleaning(text):

    text = BeautifulSoup(str(text), "html.parser").get_text()
    text=re.sub(r"[^a-zA-Z\s]", "",text)
    text=re.sub((r"^\s+|\s+|$"), " ",text)
    text=re.sub(r"@[\w\d]+", " ",text)
    text=re.sub(r"http:[\w:/\.]+", " ",text)
    text=text.lower()
    text=" ".join(text.split())
    words=[stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

if text:
    cleaned_text = data_cleaning(text)
    # converting data into vectors
    X_tfidf = vectorizer.transform([cleaned_text])

# predicting    
if st.button("🔮 Predict"):
    if text.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned_text = data_cleaning(text)
        X_tfidf = vectorizer.transform([cleaned_text])
        prediction = model.predict(X_tfidf)

        st.subheader("🧠 Prediction Result")
        if prediction[0] == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")



