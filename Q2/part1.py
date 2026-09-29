import os
import random
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import math


# Dataset paths
train_pos = "aclImdb/train/pos"
train_neg = "aclImdb/train/neg"
test_pos = "aclImdb/test/pos"
test_neg = "aclImdb/test/neg"

def load_random_reviews(folder, number):
    """
    Load a specified number of reviews from a given folder randomly.
    """
    files = os.listdir(folder)
    selected_files = random.sample(files, number)
    reviews = []

    for file in selected_files:
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            reviews.append(f.read())

    return reviews

def preprocess_reviews(reviews):
    """
    Preprocess the reviews by converting to lowercase, removing punctuation, and removing stop words.
    """
    processed_reviews = []
    stop_words = set(stopwords.words('english'))

    for review in reviews:
        # Convert to lowercase
        review = review.lower()
        # Remove punctuation
        review = review.translate(str.maketrans('', '', string.punctuation))
        # Remove stop words
        words = review.split()
        words = [word for word in words if word not in stop_words]
        # Join words back into a single string
        review = ' '.join(words)
        processed_reviews.append(review)

    return processed_reviews    

def build_vocabulary(reviews):
    word_count = Counter()

    for review in reviews:
        word_count.update(review)

    vocabulary = set(word_count.keys())

    return vocabulary

def extract_features(reviews, vocabulary):
    feature_vectors = []

    for review in reviews:
        word_count = Counter(review)

        vector = []

        for word in vocabulary:
            vector.append(word_count[word])

        feature_vectors.append(vector)

    return feature_vectors

def calculate_priors(train_pos_reviews, train_neg_reviews):
    num_positive = len(train_pos_reviews)
    num_negative = len(train_neg_reviews)

    total_reviews = num_positive + num_negative

    prior_positive = num_positive / total_reviews
    prior_negative = num_negative / total_reviews

    return prior_positive, prior_negative

def calculate_likelihoods(train_pos_tokens, train_neg_tokens, vocabulary):
    # Count words in positive and negative reviews
    positive_word_count = Counter()
    for review in train_pos_tokens:
        positive_word_count.update(review)
    negative_word_count = Counter()
    for review in train_neg_tokens:
        negative_word_count.update(review)
    # Total words in each class and vocab
    positive_total_words = sum(positive_word_count.values())
    negative_total_words = sum(negative_word_count.values())
    vocabulary_size = len(vocabulary)

    positive_likelihoods = {}
    negative_likelihoods = {}

    #calculate likelihood estimates for each word to be in each class using LaPl
    for word in vocabulary:
        positive_likelihoods[word] = (positive_word_count[word] + 1) / (positive_total_words + vocabulary_size)
        negative_likelihoods[word] = (negative_word_count[word] + 1) / (negative_total_words + vocabulary_size)
    
    return positive_likelihoods, negative_likelihoods

def classify_review(review, vocabulary, positive_likelihoods, negative_likelihoods,
                    prior_positive, prior_negative):
    word_count = Counter(review)

    #Using log here because otherwise values can get exceedingly small
    positive_score = math.log(prior_positive)
    negative_score = math.log(prior_negative)

    for word in word_count:
        #if word is in vocabulary, add its positive and negative likelihoods
        if word in vocabulary:
            count = word_count[word]
            positive_score += count * math.log(positive_likelihoods[word])
            negative_score += count * math.log(negative_likelihoods[word])

    if positive_score > negative_score:
        return 1
    else:
        return 0

def calculate_metrics (predictions, test_pos_tokens, test_neg_tokens):
    actual_labels = ([1] * len(test_pos_tokens) + [0] * len(test_neg_tokens))

    # Calculate true positive rate 
    true_positives = 0
    for i in range(len(test_pos_tokens)):
        if predictions[i] == actual_labels[i]:
                    true_positives += 1

    # Calculate false negatives
    false_negatives = len(test_pos_tokens) - true_positives

    # Calculate true negative rate
    true_negatives = 0
    for i in range(len(test_pos_tokens), len(predictions)):
        if predictions[i] == actual_labels[i]:
            true_negatives += 1

    # Calculate false positives 
    false_positives = len(test_neg_tokens) - true_negatives
    
    # Calculate accuracy, precision, recall, and F1
    accuracy = (true_positives + true_negatives) / (len(test_pos_tokens) + len(test_neg_tokens))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score, true_positives, false_negatives, false_positives, true_negatives

def main():
    # Load random reviews
    random.seed(42)
    train_pos_reviews = load_random_reviews(train_pos, 500)
    train_neg_reviews = load_random_reviews(train_neg, 500)
    test_pos_reviews = load_random_reviews(test_pos, 100)
    test_neg_reviews = load_random_reviews(test_neg, 100)

    # Preprocess reviews
    train_pos_reviews = preprocess_reviews(train_pos_reviews)
    train_neg_reviews = preprocess_reviews(train_neg_reviews)
    test_pos_reviews = preprocess_reviews(test_pos_reviews)
    test_neg_reviews = preprocess_reviews(test_neg_reviews)

    # Tokenize reviews
    train_pos_tokens = [word_tokenize(review) for review in train_pos_reviews]
    train_neg_tokens = [word_tokenize(review) for review in train_neg_reviews]
    test_pos_tokens = [word_tokenize(review) for review in test_pos_reviews]
    test_neg_tokens = [word_tokenize(review) for review in test_neg_reviews]

    #Combine train tokens
    train_tokens = train_pos_tokens + train_neg_tokens

    #Build vocab and extract features
    vocabulary = build_vocabulary(train_tokens)
    train_features = extract_features(train_tokens, vocabulary)

    prior_positive, prior_negative = calculate_priors(train_pos_reviews, train_neg_reviews)
    positive_likelihoods, negative_likelihoods = calculate_likelihoods(train_pos_tokens, train_neg_tokens, vocabulary)
    
    test_tokens = test_pos_tokens + test_neg_tokens
    predictions = []

    for review in test_tokens:
        prediction = classify_review(review, vocabulary, positive_likelihoods,
                                    negative_likelihoods, prior_positive, prior_negative
                                )
        predictions.append(prediction)

    accuracy, precision, recall, f1_score, tp, fn, fp, tn = calculate_metrics (predictions, test_pos_tokens, test_neg_tokens)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              Positive  Negative")
    print("Actual Positive   ", tp, "      ", fn)
    print("Actual Negative   ", fp, "      ", tn)

main()