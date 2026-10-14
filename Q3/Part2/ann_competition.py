import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


# 1. LOAD DATA
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_df = train_df.dropna().reset_index(drop=True)
    test_df  = test_df.dropna().reset_index(drop=True)

    print(f"Train samples : {len(train_df)}")
    print(f"Test  samples : {len(test_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}\n")
    print(f"Test  label distribution:\n{test_df['label'].value_counts()}\n")

    return train_df, test_df


# 2. TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    text = text.strip()
    return text


# 3. PREPROCESS
def preprocess(df):
    df = df.copy()
    df['cleaned'] = df['sentence'].apply(clean_text)
    return df


# 4. TF-IDF VECTORIZATION (unigram + bigram)
def tfidf_vectorize(train_df, test_df):
    vectorizer = TfidfVectorizer(
        ngram_range    = (1, 2),      
        min_df         = 2,          
        max_df         = 0.95,       
        sublinear_tf   = True,       
        strip_accents  = 'unicode',
        analyzer       = 'word',
    )

    X_train = vectorizer.fit_transform(train_df['cleaned'])
    X_test  = vectorizer.transform(test_df['cleaned'])

    print(f"TF-IDF vocab size : {len(vectorizer.vocabulary_)}")
    print(f"X_train shape     : {X_train.shape}")
    print(f"X_test  shape     : {X_test.shape}\n")

    return X_train, X_test, vectorizer


# 5. SCALE FEATURES
def scale_features(X_train, X_test):
    scaler  = MaxAbsScaler()         
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, scaler


# 6. BUILD AND TRAIN ANN
def train(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes  = (512, 256, 128),  
        activation          = 'relu',
        solver              = 'adam',
        alpha               = 1e-4,           
        batch_size          = 64,
        learning_rate       = 'adaptive',     
        learning_rate_init  = 1e-3,
        max_iter            = 500,
        random_state        = 42,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 15,              
        verbose             = True,
    )

    print("Training started...\n")
    model.fit(X_train, y_train)
    print(f"\nTraining complete! Stopped at iteration: {model.n_iter_}")

    return model


# 7. PREDICT
def predict(model, X_test):
    return model.predict(X_test)


# 8. EVALUATE
def evaluate(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    accuracy  = accuracy_score(y_true, y_pred) * 100
    cm        = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    precision_pos = precision_score(y_true, y_pred, pos_label=1)
    recall_pos    = recall_score(y_true, y_pred, pos_label=1)
    f1_pos        = f1_score(y_true, y_pred, pos_label=1)

    precision_neg = precision_score(y_true, y_pred, pos_label=0)
    recall_neg    = recall_score(y_true, y_pred, pos_label=0)
    f1_neg        = f1_score(y_true, y_pred, pos_label=0)

    macro_f1  = f1_score(y_true, y_pred, average='macro')

    print(f"\n{'='*40}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy  : {accuracy:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"           Predicted 0   Predicted 1")
    print(f"Actual 0 :     {TN}          {FP}")
    print(f"Actual 1 :     {FN}          {TP}")
    print(f"\nClass 1 (Positive):")
    print(f"  Precision : {precision_pos:.4f}")
    print(f"  Recall    : {recall_pos:.4f}")
    print(f"  F1 Score  : {f1_pos:.4f}")
    print(f"\nClass 0 (Negative):")
    print(f"  Precision : {precision_neg:.4f}")
    print(f"  Recall    : {recall_neg:.4f}")
    print(f"  F1 Score  : {f1_neg:.4f}")
    print(f"\nMacro F1  : {macro_f1:.4f}")
    print(f"{'='*40}")


# MAIN
if __name__ == "__main__":
    import re

    train_df, test_df = load_data('../Data/sentiment_train_dataset.csv',
                                   '../Data/sentiment_test_dataset.csv')

    def avg_word_index(sentence):
        nums = [int(x) for x in re.findall(r'word_(\d+)', sentence)]
        return np.mean(nums) if nums else 0

    train_df['avg_idx'] = train_df['sentence'].apply(avg_word_index)
    print(train_df.groupby('label')['avg_idx'].mean())