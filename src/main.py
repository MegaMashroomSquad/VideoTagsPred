from src.data_processing.data_loader import load_data
from src.data_processing.text_cleaning import clean_text
from src.llm_integration.llama_processor import process_title_with_llama
from src.llm_integration.prompt_templates import title_prompt_template, description_prompt_template
from src.model_training.feature_engineering import generate_features
from src.model_training.train_catboost import train_and_evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    # Загрузка и подготовка основного датасета
    df = load_data('data/train_data_categories.csv')

    # df = df.iloc[:100]
    
    # Очистка текста
    df['splitted_tags'] = df['tags'].apply(lambda x: x.split(', '))
    df['cleaned_title'] = df['title'].apply(clean_text)
    df['cleaned_description'] = df['description'].apply(clean_text)
    
    print('Началась обработка title llm')
    title_tags = []
    for title in tqdm(df['title'].tolist()):
        title_tags.append(process_title_with_llama(title, title_prompt_template))

    df['llm_title_tags'] = title_tags
    del title_tags


    print('Началась обработка description llm')
    description_tags = []
    for description in tqdm(df['description'].tolist()):
        description_tags.append(process_title_with_llama(description, title_prompt_template))

    df['llm_description_tags'] = description_tags
    del description_tags

    # Подготовка данных для генерации признаков
    df['llm_title_tags_list'] = df['llm_title_tags'].apply(lambda x: x.split(', '))
    df['llm_description_tags_list'] = df['llm_description_tags'].apply(lambda x: x.split(', '))
    
    llm_data = pd.concat([
        df['llm_title_tags_list'].str.join('|').str.get_dummies(),
        df['llm_description_tags_list'].str.join('|').str.get_dummies()
    ], axis=1)

    # Преобразование меток в формат мультилейбел с использованием MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['splitted_tags'])

    # Разделение данных на обучающую и валидационную выборки
    train_df, valid_df, y_train, y_valid = train_test_split(df, y, test_size=0.2, random_state=50)

    # Генерация признаков для обучения
    all_features = generate_features(df, llm_data)

    train_features = all_features[train_df.index]
    valid_features = all_features[valid_df.index]

    # train_features = generate_features(train_df, llm_data.loc[train_df.index])
    # valid_features = generate_features(valid_df, llm_data.loc[valid_df.index])

    # Обучение и оценка модели на валидационных данных
    ovr_estimators = train_and_evaluate_model(train_features, y_train, valid_features, y_valid)

    # Обработка другого датасета
    process_additional_dataset(ovr_estimators, mlb)

def process_additional_dataset(ovr_estimators, mlb):
    # Загрузка и подготовка дополнительного датасета
    df_additional = load_data('data/test_dataset.csv')
    
    # Очистка текста
    df_additional['cleaned_title'] = df_additional['title'].apply(clean_text)
    df_additional['cleaned_description'] = df_additional['description'].apply(clean_text)

    print('Началась обработка title llm')
    title_tags = []
    for title in tqdm(df_additional['title'].tolist()):
        title_tags.append(process_title_with_llama(title, title_prompt_template))

    df_additional['llm_title_tags'] = title_tags
    del title_tags


    print('Началась обработка description llm')
    description_tags = []
    for description in tqdm(df_additional['description'].tolist()):
        description_tags.append(process_title_with_llama(description, title_prompt_template))

    df_additional['llm_description_tags'] = description_tags
    del description_tags
    
    # # Применение LLM для обработки заголовков и описаний
    # df_additional['llm_title_tags'] = df_additional['title'].apply(lambda title: process_title_with_llama(title, title_prompt_template))
    # df_additional['llm_description_tags'] = df_additional['description'].apply(lambda description: process_title_with_llama(description, description_prompt_template))

    # Подготовка данных для генерации признаков
    df_additional['llm_title_tags_list'] = df_additional['llm_title_tags'].apply(lambda x: x.split(', '))
    df_additional['llm_description_tags_list'] = df_additional['llm_description_tags'].apply(lambda x: x.split(', '))
    
    llm_data_additional = pd.concat([
        df_additional['llm_title_tags_list'].str.join('|').str.get_dummies(),
        df_additional['llm_description_tags_list'].str.join('|').str.get_dummies()
    ], axis=1)

    # Генерация признаков для дополнительного датасета
    additional_features = generate_features(df_additional, llm_data_additional)

    # Предсказание на тестовых данных
    y_pred_proba = np.zeros(np.zeros((df_additional.shape[0], len(ovr_estimators))))

    for i, model in enumerate(ovr_estimators):
        if model is not None:
            y_pred_proba[:, i] = model.predict_proba(additional_features)[:, 1]


    y_pred = (y_pred_proba > 0.2).astype(np.uint8)

    bad_ids = y_pred.sum(axis=1)==0
    for i in range(len(y_pred)):
        if bad_ids[i] and y_pred_proba.max(axis=1)[i] > 0.05:
            print(i)
            y_pred[i, y_pred_proba.argmax(axis=1)[i]] = 1

    # Добавление предсказанных тегов в датафрейм
    df_additional['predicted_tags'] = mlb.inverse_transform(y_pred)

    # Сохранение результатов
    df_additional.to_csv('data/other_dataset_predictions.csv', index=False)


if __name__ == "__main__":
    main()
