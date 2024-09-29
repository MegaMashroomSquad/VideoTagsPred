import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_features(df, llm_data):
    """
    Генерирует признаки на основе заголовков, описаний и данных, полученных от LLM.
    
    Аргументы:
    - df: DataFrame, содержащий данные для генерации признаков.
    - llm_data: DataFrame, содержащий one-hot закодированные данные из LLM.
    
    Возвращает:
    - features: numpy массив сгенерированных признаков.
    """

    # Генерация TF-IDF признаков для заголовков и описаний
    print(df['cleaned_title'])
    tfidf_title = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=350,
        min_df=5,
    )
    X_title_tfidf = tfidf_title.fit_transform(df['cleaned_title']).toarray()
    
    print(df['cleaned_description'])
    tfidf_description = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=150,
        min_df=5,
    )
    X_description_tfidf = tfidf_description.fit_transform(df['cleaned_description']).toarray()
    
    # Объединение TF-IDF признаков и данных, полученных от LLM
    features = np.hstack((X_title_tfidf, X_description_tfidf, llm_data.values))
    
    return features