import streamlit as st
import joblib
import os

# Set page config for a better UI look
st.set_page_config(
    page_title="News Headline Sentiment",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .sentiment-positive {
        color: #155724;
        background-color: #d4edda;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        font-weight: 500;
    }
    .sentiment-negative {
        color: #721c24;
        background-color: #f8d7da;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        font-weight: 500;
    }
    .sentiment-neutral {
        color: #383d41;
        background-color: #e2e3e5;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #6c757d;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model_path = 'models/sentiment_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    else:
        return None, None

def main():
    st.title("📰 News Headline Sentiment Analysis")
    st.markdown("Predict the emotional tone of news headlines (**Positive**, **Neutral**, **Negative**).")
    
    st.divider()
    
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.info("Training sentiment model on the fly...")
        import train_sentiment
        train_sentiment.train_and_save()
        model, vectorizer = load_models()
    
    # User Input
    headline_input = st.text_area("Enter a news headline:", height=100, placeholder="Example: The company reported a 20% increase in revenue for Q3.")
    
    if st.button("Predict Sentiment", type="primary", use_container_width=True):
        if headline_input.strip() == "":
            st.warning("Please enter a valid headline.")
        else:
            with st.spinner("Analyzing..."):
                input_tfidf = vectorizer.transform([headline_input])
                prediction = model.predict(input_tfidf)[0]
                
                st.subheader("Result:")
                if prediction == "Positive":
                    st.success("✅ **Positive** Sentiment")
                    st.markdown(f'<div class="sentiment-positive">"{headline_input}"</div>', unsafe_allow_html=True)
                elif prediction == "Negative":
                    st.error("📉 **Negative** Sentiment")
                    st.markdown(f'<div class="sentiment-negative">"{headline_input}"</div>', unsafe_allow_html=True)
                else:
                    st.info("⚖️ **Neutral** Sentiment")
                    st.markdown(f'<div class="sentiment-neutral">"{headline_input}"</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
