import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_features(df, llm_data):
    # Разделение текстов на обучающую и тестовую выборки
    X_title = df['cleaned_title']
    X_description = df['cleaned_description']
    
    # Генерация признаков с использованием TF-IDF
    tfidf_title = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=350,
        min_df=5,
    )
    X_title_tfidf = tfidf_title.fit_transform(X_title).toarray()
    
    tfidf_description = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=150,
        min_df=5,
    )
    X_description_tfidf = tfidf_description.fit_transform(X_description).toarray()
    
    # Объединение признаков (TF-IDF + данные из LLM)
    features = np.hstack((X_title_tfidf, llm_data.values))
    
    # Преобразование меток (labels)
    labels = df['first_level_list'].apply(lambda x: x.split(', '))
    
    return features, labels