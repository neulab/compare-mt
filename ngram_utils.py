from collections import defaultdict
import itertools

def sent_ngrams_list(words, n):
  """
  Create a list with all the n-grams in a sentence

  Arguments:
    words: A list of strings representing a sentence
    n: The ngram length to consider

  Returns:
    A list of n-grams in the sentence
  """
  word_ngram = []
  for i in range(len(words) - n + 1):
    ngram = tuple(words[i:i + n])
    word_ngram.append(ngram)
  return word_ngram

def iterate_sent_ngrams(words, labels=None, min_length=1, max_length=4):
  """
  Create a list with all the n-grams in a sentence

  Arguments:
    words: A list of strings representing a sentence
    labels: A list of labels on each word in the sentence, optional (will use `words` if not specified)
    min_length: The minimum ngram length to consider
    max_length: The maximum ngram length to consider

  Returns:
    An iterator over n-grams in the sentence with both words and labels
  """
  if labels is not None and len(labels) != len(words):
    raise ValueError(f'length of labels and sentence must be the same but got'
                     f' {len(words)} != {len(labels)} at\n{words}\n{labels}')
  for n in range(min_length-1, max_length):
    for i in range(len(words) - n):
      word_ngram = tuple(words[i:i + n + 1])
      label_ngram = tuple(labels[i:i + n + 1]) if (labels is not None) else word_ngram
      yield word_ngram, label_ngram

def compare_ngrams(ref, out, ref_labels=None, out_labels=None, min_length=1, max_length=4):
  """
  Compare n-grams appearing in the reference sentences and output

  Args:
    ref: A list of reference sentences
    out: A list of output sentences
    ref_labels: Alternative labels for reference words (e.g. POS tags) to use when aggregating counts
    out_labels: Alternative labels for output words (e.g. POS tags) to use when aggregating counts
    min_length: The minimum length of n-grams to consider
    max_length: The maximum length of n-grams to consider

  Returns:
    A tuple of dictionaries including
      total: the total number of n-grams in the output
      match: the total number of matched n-grams appearing in both output and reference
      over: the total number of over-generated n-grams appearing in output but not reference
      under: the total number of under-generated n-grams appearing in output but not reference
  """
  if (ref_labels is None) != (out_labels is None):
    raise ValueError('ref_labels or out_labels must both be either None or not None')
  total, match, over, under = [defaultdict(lambda: 0) for _ in range(4)]
  if ref_labels is None: ref_labels = []
  if out_labels is None: out_labels = []
  for ref_sent, out_sent, ref_lab, out_lab in itertools.zip_longest(ref, out, ref_labels, out_labels):
    # Find the number of reference n-grams (on a word level)
    ref_ngrams = list(iterate_sent_ngrams(ref_sent, labels=ref_lab, min_length=min_length, max_length=max_length))
    ref_word_counts = defaultdict(lambda: 0)
    for ref_w, ref_l in ref_ngrams:
      ref_word_counts[ref_w] += 1
    # Step through the output ngrams and find matched and overproduced ones
    for out_w, out_l in iterate_sent_ngrams(out_sent, labels=out_lab, min_length=min_length, max_length=max_length):
      total[out_l] += 1
      if ref_word_counts[out_w] > 0:
        match[out_l] += 1
        ref_word_counts[out_w] -= 1
      else:
        over[out_l] += 1
    # Remaining ones are underproduced
    # (do reverse order just to make ordering consistent for over and under, shouldn't matter much)
    for ref_w, ref_l in reversed(ref_ngrams):
      if ref_word_counts[ref_w] > 0:
        under[ref_l] += 1
        ref_word_counts[ref_w] -= 1
  return total, match, over, under
