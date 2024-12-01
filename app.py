import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

# Sample pre-trained model and vectorizer (for demonstration purposes)
# Replace this with your actual model and vectorizer
vectorizer = CountVectorizer()
model = GaussianNB()

# Predefined data to simulate training (replace with your actual dataset during deployment)
sample_reviews = ["great book loved it", "terrible waste of time", "amazing storyline", "not worth the money"]
sample_labels = [1, 0, 1, 0]

# Simulating model training
X_train = vectorizer.fit_transform(sample_reviews).toarray()
model.fit(X_train, sample_labels)

# Page 1: Welcome Page
def welcome_page():
    st.title("📖 Welcome to the Kindle Review Sentiment Analyzer!")
    st.markdown(
        """
        ### Discover the sentiment of your book reviews!
        Input your review and let our AI predict whether it's **Positive** or **Negative**.
        **Click below to proceed!**
        """
    )
    if st.button("✨ Go to Prediction Page"):
        st.session_state["page"] = "prediction"

# Page 2: Prediction Page
def prediction_page():
    st.title("📝 Input Your Review")
    st.markdown(
        """
        ### Enter the review text below:
        """
    )
    user_review = st.text_area("Your Review:", placeholder="Type your review here...")
    
    if st.button("🔍 Analyze Review"):
        if user_review.strip():
            # Transform input review
            review_vector = vectorizer.transform([user_review]).toarray()
            # Predict sentiment
            prediction = model.predict(review_vector)[0]
            sentiment = "Positive 😄" if prediction == 1 else "Negative 😞"
            st.session_state["result"] = sentiment
            st.session_state["page"] = "result"
        else:
            st.error("Please enter a valid review.")

# Page 3: Result Page
def result_page():
    st.title("📊 Sentiment Analysis Result")
    result = st.session_state.get("result", "Unknown")
    st.subheader(f"The sentiment of your review is: **{result}**")
    
    if "Positive" in result:
        st.success("🎉 Great! The review is positive.")
    else:
        st.warning("⚠️ The review seems negative. Consider addressing the issues.")
    
    if st.button("🏠 Back to Home"):
        st.session_state["page"] = "welcome"

# App Navigation Logic
if "page" not in st.session_state:
    st.session_state["page"] = "welcome"

page = st.session_state["page"]
if page == "welcome":
    welcome_page()
elif page == "prediction":
    prediction_page()
elif page == "result":
    result_page()
