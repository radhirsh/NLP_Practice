import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

def train_and_save_model():
    # Download stopwords if not present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Load dataset (adjust the path if needed)
    df = pd.read_csv(r'C:\Users\Acuvate\OneDrive\NLP\Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

    # Clean and preprocess the reviews
    def cleaning():
        corpus = []
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        for i in range(len(df)):
            review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
            review = review.lower()
            words = review.split()
            words = [w for w in words if w not in stop_words]
            words = [ps.stem(w) for w in words]
            cleaned_review = ' '.join(words)
            corpus.append(cleaned_review)
        return corpus

    corpus = cleaning()

    # Create Bag of Words model
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, 1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    # Train Multinomial Naive Bayes
    classifier = MultinomialNB(alpha=0.2)
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")

    # Save model and vectorizer
    joblib.dump(classifier, 'sentiment_model.pkl')
    joblib.dump(cv, 'count_vectorizer.pkl')

    print("Model and vectorizer saved as 'sentiment_model.pkl' and 'count_vectorizer.pkl'")

