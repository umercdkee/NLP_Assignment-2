import os
import re
import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

train_pos = "aclImdb/train/pos"
train_neg = "aclImdb/train/neg"
test_pos  = "aclImdb/test/pos"
test_neg  = "aclImdb/test/neg"

def load_all_reviews(folder):
    reviews = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            reviews.append(f.read())
    return reviews

NEGATION_WORDS = {'not', 'no', 'never', 'nor', 'neither', 'nobody', 'nothing', 'nowhere', "n't"}
RESET_PUNCT = re.compile(r'[.!?,;:]')

def handle_negation(text):
    tokens = text.lower().split()
    result = []
    negated = False
    for token in tokens:
        if RESET_PUNCT.search(token):
            negated = False
        clean = re.sub(r"[^a-z']", '', token)
        if clean in NEGATION_WORDS or token.endswith("n't"):
            negated = True
            result.append(token)
        else:
            result.append('NOT_' + token if negated else token)
    return ' '.join(result)

def preprocess(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = handle_negation(text)
    text = re.sub(r"[^\w\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate(y_true, y_pred):
    acc      = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred)
    rec      = recall_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm       = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'='*40}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred Pos  Pred Neg")
    print(f"Actual Pos    {tp:6d}    {fn:6d}")
    print(f"Actual Neg    {fp:6d}    {tn:6d}")
    print(f"{'='*40}")

if __name__ == "__main__":
    print("Loading reviews...")
    train_pos_rev = load_all_reviews(train_pos)
    train_neg_rev = load_all_reviews(train_neg)
    test_pos_rev  = load_all_reviews(test_pos)
    test_neg_rev  = load_all_reviews(test_neg)

    y_train = [1]*len(train_pos_rev) + [0]*len(train_neg_rev)
    y_test  = [1]*len(test_pos_rev)  + [0]*len(test_neg_rev)

    print("Preprocessing...")
    train_texts = [preprocess(r) for r in train_pos_rev + train_neg_rev]
    test_texts  = [preprocess(r) for r in test_pos_rev  + test_neg_rev]

    print("Vectorizing (TF-IDF, trigrams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.95, sublinear_tf=True)
    X_train_full = vectorizer.fit_transform(train_texts) 
    X_test_full  = vectorizer.transform(test_texts)
    print(f"Vocab size before selection: {X_train_full.shape[1]}")

    print("Tuning k and alpha via 5-fold CV on training data...")
    print(f"{'k':>8}  {'alpha':>6}  {'CV Macro F1':>12}")
    print("-" * 35)

    best_score = 0
    best_config = {}

    for k in [40000, 60000, 80000, 100000]:
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
            pipe = Pipeline([
                ('selector', SelectKBest(chi2, k=k)),
                ('model', ComplementNB(alpha=alpha))
            ])
            scores = cross_val_score(pipe, X_train_full, y_train, cv=5, scoring='f1_macro')
            mean_score = scores.mean()
            marker = " ← BEST" if mean_score > best_score else ""
            print(f"{k:>8}  {alpha:>6.2f}  {mean_score:>12.4f}{marker}")
            if mean_score > best_score:
                best_score = mean_score
                best_config = {'k': k, 'alpha': alpha}

    print(f"\nBest via CV: k={best_config['k']}, alpha={best_config['alpha']}")

    print(f"Feature selection (chi2, top {best_config['k']})...")
    selector = SelectKBest(chi2, k=best_config['k'])
    X_train = selector.fit_transform(X_train_full, y_train)
    X_test  = selector.transform(X_test_full)

    print("Training ComplementNB...")
    model = ComplementNB(alpha=best_config['alpha'])
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)