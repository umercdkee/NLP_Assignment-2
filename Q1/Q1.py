from collections import Counter
import math
import re

#load and preprocess data
def load_and_preprocess(filepath):
  with open(filepath, "r", encoding="utf-8") as file:
    text = file.read().lower()
  text = re.sub(r"[^a-z0-9\s]", "", text)
  tokens = text.split()
  return tokens

#generate n-grams and calculate probabilities
def generate_ngrams(tokens, n):
  return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

#build vocab and count n-grams and n-1 gram contexts
def train_ngram_model(train_tokens, n):
  vocab = set(train_tokens)
  ngram_counts = Counter(generate_ngrams(train_tokens, n))
  context_counts = (
      Counter(generate_ngrams(train_tokens, n - 1))
      if n > 1
      else len(train_tokens))
  return vocab, ngram_counts, context_counts

#calc prob of word using n-1 words using add k 
def calculate_probability(ngram, n, vocab_size, ngram_counts, context_counts, k=0.1):
  ngram_count = ngram_counts[ngram]
  if n == 1:
    total_count = context_counts  #total tokens in training data
    return (ngram_count + k) / (total_count + k * vocab_size)
  else:
    context = ngram[:-1]
    context_count = context_counts[context]
    return (ngram_count + k) / (context_count + k * vocab_size)

def calculate_perplexity(test_tokens, n, vocab, ngram_counts, context_counts, k=0.1):
  test_ngrams = generate_ngrams(test_tokens, n)
  vocab_size = len(vocab)
  M = len(test_ngrams)
  log_prob_sum = 0.0
  for ngram in test_ngrams:
    prob = calculate_probability(ngram, n, vocab_size, ngram_counts, context_counts, k)
    log_prob_sum += math.log(prob)
  perplexity = math.exp(-log_prob_sum / M)
  return perplexity

if __name__ == "__main__":
  train_file = "train.txt"
  test_file = "test.txt"
  train_tokens = load_and_preprocess(train_file)
  test_tokens = load_and_preprocess(test_file)
  print(f"--- Perplexity Results on Test Set - Add-k Smoothing (k = 0.1) ---")
  for n, name in [(1, "Unigram"), (2, "Bigram"), (3, "Trigram")]:
    vocab, ngram_counts, context_counts = train_ngram_model(train_tokens, n)
    pp = calculate_perplexity(test_tokens, n, vocab, ngram_counts, context_counts, 0.1)
    print(f"{name} (n={n}) Perplexity: {pp:.4f}")
  print()
