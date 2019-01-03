from collections import defaultdict

def calc_sent_ngrams(sent, labels=None, min_length=1, max_length=4):
  """
  Calculate the number of n-grams in a sentence

  Arguments:
    sent: A list of strings representing a sentence
    labels: A list of labels on each word in the sentence, optional
    max_length: The minimum ngram length to consider
    max_length: The maximum ngram length to consider

  Returns:
    A dictionary with all the n-grams of length min_length to max_length
  """
  if labels is not None and len(labels) != len(sent):
    raise ValueError('length of labels and sentence must be the same')
  ret = defaultdict(lambda: 0)
  for n in range(min_length-1, max_length):
    for i in range(len(sent)-n):
      if not labels:
        ret[tuple(sent[i:i+n+1])] += 1
      else:
        ret[(tuples(sent[i:i+n+1]),tuples(labels[i:i+n+1]))] += 1
  return ret

def match_dicts(left, right):
  """
  Calculate the number of overlapping items in a counter dictionary

  Args:
    left: A dictionary with counts
    right: Another dictionary with counts

  Returns:
    A dictionary with the minimum of the counts in the two dictionaries
  """
  ret = defaultdict(lambda: 0)
  for k, v in left.items():
    if k in right:
      ret[k] = min(v, right[k])
  return ret

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
  for ref_sent, out_sent in zip(ref, out):
    ref_ngrams = calc_sent_ngrams(ref_sent, labels=ref_labels, min_length=min_length, max_length=max_length)
    out_ngrams = calc_sent_ngrams(out_sent, labels=out_labels, min_length=min_length, max_length=max_length)
    all_keys = set(ref_ngrams.keys()) | set(out_ngrams.keys())
    for k in all_keys:
      ov, rv = out_ngrams[k], ref_ngrams[k]
      agg_k = k if not ref_labels else k[1]
      total[agg_k] += ov
      match[agg_k] += min(rv, ov)
      if ov > rv:
        over[agg_k] += ov - rv
      if rv > ov:
        over[agg_k] += rv - ov
  return total, match, over, under