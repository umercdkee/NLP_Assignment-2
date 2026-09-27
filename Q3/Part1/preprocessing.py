import pandas as pd
import numpy as np
import re

#Load Data
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    
    print(f"Train samples : {len(train_df)}")
    print(f"Test  samples : {len(test_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}\n")
    print(f"Test  label distribution:\n{test_df['label'].value_counts()}\n")
    
    return train_df, test_df

#Text Cleaning
def clean_text(text):
    text = text.lower()                        
    text = re.sub(r'[^a-z0-9\s_]', '', text)  
    text = text.strip()
    return text

#Tokenization
def tokenize(text):
    return text.split()

#Preprocessing
def preprocess(df):
    df = df.copy()
    df['tokens'] = df['sentence'].apply(lambda x: tokenize(clean_text(x)))
    return df

#Build Vocabulary
def build_vocabulary(train_df):
    vocab = {}
    idx = 0
    for tokens in train_df['tokens']:
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    
    print(f"Vocabulary size: {len(vocab)}\n")
    return vocab

#One-hot Encoding
def one_hot_encode(df, vocab):
    vocab_size = len(vocab)
    num_samples = len(df)
    
    X = np.zeros((num_samples, vocab_size), dtype=np.float32)
    
    for i, tokens in enumerate(df['tokens']):
        for token in tokens:
            if token in vocab:
                X[i, vocab[token]] = 1.0
    
    return X

if __name__ == "__main__":
     
    train_df, test_df = load_data('../Data/sentiment_train_dataset.csv',
                               '../Data/sentiment_test_dataset.csv')
    
    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)
    
    vocab = build_vocabulary(train_df)
    
    X_train = one_hot_encode(train_df, vocab)
    X_test  = one_hot_encode(test_df,  vocab)
    
    y_train = train_df['label'].values.reshape(-1, 1).astype(np.float32)
    y_test  = test_df['label'].values.reshape(-1, 1).astype(np.float32)
    
    #Save Files
    np.save('X_train.npy', X_train)
    np.save('X_test.npy',  X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy',  y_test)
    np.save('vocab.npy',   vocab)
    
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test  shape : {y_test.shape}")
    print("\nAll files saved successfully!")