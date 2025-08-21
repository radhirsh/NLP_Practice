from django.shortcuts import render
from django.http import HttpResponse
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from  .models import Review_DB
# Import your training helper
from .helpers import train_and_save_model

# Load model & vectorizer once at module load to avoid repeated loads per request
try:
    classifier = joblib.load('sentiment_model.pkl')  # Adjust path if needed
    cv = joblib.load('count_vectorizer.pkl')        # Adjust path if needed
except FileNotFoundError:
    classifier = None
    cv = None


def retrain_model_view():
    """
    Call this view to retrain your model and save new pkl files.
    """
    train_and_save_model()

    # Reload the model and vectorizer after training
    global classifier, cv
    classifier = joblib.load('sentiment_model.pkl')
    cv = joblib.load('count_vectorizer.pkl')

    return HttpResponse("Model retrained and saved successfully!")


def preprocess_text(text):
    """
    Clean and preprocess input text for prediction.
    """
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    return ' '.join(words)


def predict_sentiment(text):
    """
    Predict sentiment of given text using the trained model.
    """
    if classifier is None or cv is None:
        return "Model is not trained yet. Please retrain first."

    processed = preprocess_text(text)
    vect = cv.transform([processed]).toarray()
    pred = classifier.predict(vect)[0]
    return "Positive" if pred == 1 else "Negative"


def index(request):
    """
    Main page to input review and get sentiment prediction.
    """
    prediction = None
    review = ""

    if request.method == "POST":
        review = request.POST.get("review")

        retrain_model_view()
        if review:
            prediction = predict_sentiment(review)
            Review_DB.objects.create(User_query=review, classification=prediction)
          


    return render(request, "index.html", {
        "prediction": prediction,
        "review": review,
        'Review_DB':Review_DB.objects.all()
    })
