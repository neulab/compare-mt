import itertools
from collections import defaultdict

import corpus_utils

class WordBucketer:

  def calc_bucketed_matches(self, ref, out):
    """
    Calculate the number of matches, bucketed by the type of word we have
    This must be used with a subclass that has self.bucket_strs defined, and self.calc_bucket(word) implemented.

    Args:
      ref: The reference corpus
      out: The output corpus

    Returns:
      A tuple containing:
        both_tot: the frequency of a particular bucket appearing in both output and reference
        ref_tot: the frequency of a particular bucket appearing in just reference
        out_tot: the frequency of a particular bucket appearing in just output
        rec: recall of the bucket
        prec: precision of the bucket
        fmeas: f1-measure of the bucket
    """
    matches = [[0, 0, 0] for x in self.bucket_strs]
    for ref_sent, out_sent in zip(ref, out):
      ref_freq, out_freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
      for x in ref_sent:
        ref_freq[x] += 1
      for x in out_sent:
        out_freq[x] += 1
      for word in set(itertools.chain(ref_freq.keys(), out_freq.keys())):
        bucket = self.calc_bucket(word)
        matches[bucket][0] += min(ref_freq[word], out_freq[word])
        matches[bucket][1] += ref_freq[word]
        matches[bucket][2] += out_freq[word]
    for both_tot, ref_tot, out_tot in matches:
      if both_tot == 0:
        rec, prec, fmeas = 0.0, 0.0, 0.0
      else:
        rec = both_tot / float(ref_tot)
        prec = both_tot / float(out_tot)
        fmeas = 2 * prec * rec / (prec + rec)
      yield both_tot, ref_tot, out_tot, rec, prec, fmeas


class FreqWordBucketer(WordBucketer):

  def __init__(self,
               freq_counts=None, freq_count_file=None, freq_corpus_file=None, freq_data=None,
               bucket_cutoffs=None):
    """
    A bucketer that buckets words by their frequency.

    Args:
      freq_counts: A dictionary containing word/count data.
      freq_count_file: A file containing counts for each word in tab-separated word, count format.
                       Ignored if freq_counts exists.
      freq_corpus_file: A file with a corpus used for collecting counts. Ignored if freq_count_file exists.
      freq_data: A tokenized corpus from which counts can be calculated. Ignored if freq_corpus_file exists.
      bucket_cutoffs: Cutoffs for each bucket.
                      The first bucket will be range(0,bucket_cutoffs[0]).
                      Middle buckets will be range(bucket_cutoffs[i],bucket_cutoffs[i-1].
                      Final bucket will be everything greater than bucket_cutoffs[-1].
    """
    if not freq_counts:
      freq_counts = defaultdict(lambda: 0)
      if freq_count_file != None:
        with open(freq_count_file, "r") as f:
          for line in f:
            word, freq = line.strip().split('\t')
            freq_counts[word] = freq
      elif freq_corpus_file:
        for word in itertools.chain(corpus_utils.iterate_tokens(freq_corpus_file)):
          freq_counts[word] += 1
      elif freq_data:
        for words in freq_data:
          for word in words:
            freq_counts[word] += 1
      else:
        raise ValueError('Must have at least one source of frequency counts for FreqWordBucketer')
    self.freq_counts = freq_counts

    if bucket_cutoffs is None:
      bucket_cutoffs = [1, 2, 3, 4, 5, 10, 100, 1000]
    self.bucket_cutoffs = bucket_cutoffs
    self.bucket_strs = []
    last_start = 0
    for x in bucket_cutoffs:
      if x-1 == last_start:
        self.bucket_strs.append(str(last_start))
      else:
        self.bucket_strs.append("{}-{}".format(last_start, x-1))
      last_start = x
    self.bucket_strs.append("{}+".format(last_start))

  def calc_bucket(self, word):
    freq = self.freq_counts.get(word, 0)
    for i, v in enumerate(self.bucket_cutoffs):
      if freq < v:
        return i
    return len(self.bucket_cutoffs)

  def name(self):
    return "frequency"

def create_bucketer_from_profile(bucket_type,
                                 freq_counts=None, freq_count_file=None, freq_corpus_file=None, freq_data=None,
                                 bucket_cutoffs=None):
  if bucket_type == 'freq':
    return FreqWordBucketer(
      freq_counts=freq_counts,
      freq_count_file=freq_count_file,
      freq_corpus_file=freq_corpus_file,
      freq_data=freq_data,
      bucket_cutoffs=bucket_cutoffs)
  else:
    raise ValueError(f'Illegal bucket type {bucket_type}')
