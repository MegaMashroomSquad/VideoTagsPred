from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def fit_ovr_with_eval_set(X_train, y_train, X_valid, y_valid):
    n_classes = y_train.shape[1]
    estimators = []
    
    for i in range(n_classes):
        if y_train.sum(axis=0)[i] < 3 or y_valid.sum(axis=0)[i] == 0:
            estimators.append(None)
            continue

        print(f"Training model for class {i + 1}/{n_classes}")
        model = CatBoostClassifier(
            iterations=3500, 
            learning_rate=0.03,
            depth=3,
            verbose=100,
            random_seed=42,
            early_stopping_rounds=1000,
        )
        model.fit(
            X_train, y_train[:, i],
            eval_set=(X_valid, y_valid[:, i])
        )
        estimators.append(model)
    
    return estimators

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    ovr_estimators = fit_ovr_with_eval_set(X_train, y_train, X_test, y_test)
    
        # Предсказание на тестовых данных
    y_pred_proba = np.zeros(y_test.shape)

    for i, model in enumerate(ovr_estimators):
        if model is not None:
            y_pred_proba[:, i] = model.predict_proba(X_test)[:, 1]


    y_pred = (y_pred_proba > 0.2).astype(np.uint8)

    # Вычисление accuracy и F1-меры для мультилейбел данных
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 micro: {f1_micro:.4f}')
    print(f'F1 macro: {f1_macro:.4f}')

    bad_ids = y_pred.sum(axis=1)==0
    for i in range(len(y_pred)):
        if bad_ids[i] and y_pred_proba.max(axis=1)[i] > 0.05:
            print(i)
            y_pred[i, y_pred_proba.argmax(axis=1)[i]] = 1

    # Вычисление accuracy и F1-меры
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 micro: {f1_micro:.4f}')
    print(f'F1 macro: {f1_macro:.4f}')
    
    return ovr_estimators
