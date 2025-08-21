

## 📘 SentimentScope: Real-Time Emotion Analyzer

````markdown
# 🤖 SentimentScope: Real-Time Emotion Analyzer

Welcome to **SentimentScope**, a web-based NLP application that predicts sentiment (Positive / Negative) from user-submitted text in real-time using Natural Language Processing and Machine Learning.

---

## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [🚀 Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📦 Installation](#-installation)
- [🧠 NLP Pipeline](#-nlp-pipeline)
- [📸 Screenshots](#-screenshots)
- [📚 Certification](#-certification)
- [📌 Future Scope](#-future-scope)
- [📬 Contact](#-contact)

---

## 🔍 Overview

**SentimentScope** is a chatbot-style Django application that accepts user input, applies NLP preprocessing, and classifies the sentiment using a pre-trained machine learning model.

- ✅ Real-time text prediction
- 🧠 NLP pipeline: preprocessing, tokenization, stemming, stopword removal
- 💬 Recent history tracking with emojis
- 🧪 Retrain model button (optional)
- 💽 SQLite DB for storing queries and results

---

## 🚀 Features

| Feature                        | Description                                                   |
|-------------------------------|---------------------------------------------------------------|
| 🔤 Text Input                 | User can submit a sentence or review                         |
| 🧹 NLP Preprocessing          | Clean, stem, and remove stopwords                            |
| 🤖 ML Classification          | Trained using Naive Bayes or Logistic Regression             |
| 🗃️ Review History             | Sidebar shows recent classified reviews                      |
| 🎨 Chatbot UI                 | Responsive UI with Bootstrap 5 & Font Awesome                |
| ♻️ Optional Model Retraining  | Trigger model retraining via button                          |

---

## 🛠️ Tech Stack

| Layer       | Tools Used                                  |
|-------------|---------------------------------------------|
| Backend     | Django, Python                              |
| Frontend    | HTML, CSS, Bootstrap 5, Font Awesome         |
| ML/NLP      | NLTK, scikit-learn, joblib, CountVectorizer |
| Database    | SQLite (via Django ORM)                     |

---

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/sentimentscope.git
   cd sentimentscope
````

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK stopwords**

   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. **Run the app**

   ```bash
   python manage.py runserver
   ```

6. **Access it** at:

   ```
   http://127.0.0.1:8000/
   ```

---

## 🧠 NLP Pipeline

| Step           | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| 🧹 Cleaning    | Remove non-alphabet characters                                         |
| 🔡 Lowercasing | Convert all text to lowercase                                          |
| ✂️ Tokenizing  | Split sentence into words                                              |
| ❌ Stopwords    | Remove common stopwords (e.g., "is", "the", "and")                     |
| 🌱 Stemming    | Reduce words to their base/root form (e.g., "loved" → "love")          |
| 🧮 Vectorizing | Convert processed text to feature vectors using CountVectorizer        |
| 🧠 Prediction  | Use trained ML model to classify the sentiment as Positive or Negative |

---



---

## 📚 Certification

✅ Successfully completed the **Natural Language Processing (NLP)** certification from **Intellipaat**, covering:

* NLP with NLTK
* Sentiment Analysis
* Text Classification
* POS tagging, Named Entity Recognition
* Model deployment basics

---

## 📌 Future Scope

* 🔄 Add support for Neutral classification
* 📈 Live analytics dashboard (sentiment trends)
* 🌍 Deploy on Heroku / AWS / Render
* 🤖 Integrate with a chatbot or Telegram bot
* 🔐 User login + feedback saving

---

## 📬 Contact

👨‍💻 Developed by: **Sridhar S**
🔗 LinkedIn: [Sridhar.S](https://www.linkedin.com/in/sridhar-s-076337178/)
📧 Email: [sridharsukumar888@gmail.com](mailto:sridharsukumar888@gmail.com)





