from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
topic_pipeline = pipeline("zero-shot-classification")
topic_labels = ["technology", "politics", "health", "education", "entertainment", "sports"]

def analyze_text(text):
    sentiment = sentiment_pipeline(text[:512])[0]
    emotion = emotion_pipeline(text[:512])[0]
    topic = topic_pipeline(text[:512], candidate_labels=topic_labels)
    return sentiment, emotion, topic