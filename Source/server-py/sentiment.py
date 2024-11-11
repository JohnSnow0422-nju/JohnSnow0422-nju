from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from snownlp import SnowNLP
en = SentimentIntensityAnalyzer()


def get_sentiment(word, lang='en'):
    try:
        return (SnowNLP(word).sentiments - 0.5) * 2 if lang == 'zh' else en.polarity_scores(word).get('compound', 0)
    except Exception as err:
        print(err)
        return 0
