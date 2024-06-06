import streamlit as st
import pickle
import re

# Load models and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    TF_IDF_Vectorizer = pickle.load(f)
with open('naive_bayes_model.pkl', 'rb') as f:
    NaiveBayes_classifier = pickle.load(f)

def preprocess_review(review):
    review = re.sub(r'<.*?>', '', review)
    return review

def predict_sentiment(review):
    review = preprocess_review(review)
    review_vectorized = TF_IDF_Vectorizer.transform([review])
    nb_prediction = NaiveBayes_classifier.predict(review_vectorized)[0]
    return nb_prediction

def main():
    st.title("Sentiment Analysis")

    review_input = st.text_area("Enter a movie review:", "")
    
    if st.button("Predict"):
        if review_input:
            nb_prediction = predict_sentiment(review_input)
            sentiment_map = {0: 'Negative', 1: 'Positive'}
            st.write(f"Naive Bayes Prediction: {sentiment_map[nb_prediction]}")
        else:
            st.write("Please enter a review to predict.")
    
    linkedin_url = "https://www.linkedin.com/in/ahmed-elhoseiny-2a952122a"
    github_url = "https://github.com/AhmedElhoseiny"
    email = "ahmedelhoseiny20022010@gmail.com"
    
    # Sidebar with contact information
    st.sidebar.image("Ahmed.jpg", width=100)
    st.sidebar.write("Connect with me:")
    st.sidebar.markdown(f"[![Email](https://img.shields.io/badge/Email-Contact-informational)](mailto:{email})")
    st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-Profile-green)]({github_url})")
    st.sidebar.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)]({linkedin_url})")

if __name__ == '__main__':
    main()