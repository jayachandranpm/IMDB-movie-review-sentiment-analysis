import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the IMDb dataset
df = pd.read_csv('Test.csv')

# Preprocess text data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

df['review'] = df['review'].apply(preprocess_text)

# Train the model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

model = LogisticRegression()
model.fit(X, y)

# Streamlit app
st.title('Sentiment Analysis Tool')

review_text = st.text_area('Enter your movie review:')
if st.button('Analyze'):
    cleaned_text = preprocess_text(review_text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    st.write(f'The sentiment of the review is: {sentiment}')
