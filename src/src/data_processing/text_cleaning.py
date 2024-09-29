import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('russian'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
    text = ' '.join([x for x in text.split() if x not in stop_words])
    return text