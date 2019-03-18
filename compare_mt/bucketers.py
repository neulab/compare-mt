import itertools
from collections import defaultdict

from compare_mt import corpus_utils
from compare_mt import scorers
from compare_mt import arg_utils

class Bucketer:

  def set_bucket_cutoffs(self, bucket_cutoffs, num_type='int'):
    self.bucket_cutoffs = bucket_cutoffs
    self.bucket_strs = []
    for i, x in enumerate(bucket_cutoffs):
      if i == 0:
        self.bucket_strs.append(f'<{x}')
      elif num_type == 'int' and x-1 == bucket_cutoffs[i-1]:
        self.bucket_strs.append(f'{x-1}')
      else:
        self.bucket_strs.append(f'[{bucket_cutoffs[i-1]},{x})')
    self.bucket_strs.append(f'>={x}')

  def cutoff_into_bucket(self, value):
    for i, v in enumerate(self.bucket_cutoffs):
      if value < v:
        return i
    return len(self.bucket_cutoffs)

class WordBucketer(Bucketer):

  def calc_bucket(self, val, ref_label=None, out_label=None, src_label=None):
    """
    Calculate the bucket for a particular word

    Args:
      val: The word to calculate the bucket for
      ref_label: If there's a label on the reference word, add it
      out_label: If there's a label on the output word, add it
      src_label: If there's a label on the source word, add it

    Returns:
      An integer ID of the bucket
    """
    raise NotImplementedError('calc_bucket must be implemented in subclasses of WordBucketer')

  def calc_bucketed_matches(self, ref, out, ref_labels=None, out_labels=None):
    """
    Calculate the number of matches, bucketed by the type of word we have
    This must be used with a subclass that has self.bucket_strs defined, and self.calc_bucket(word) implemented.

    Args:
      ref: The reference corpus
      out: The output corpus
      ref_labels: Labels of the reference corpus (optional)
      out_labels: Labels of the output corpus (should be specified iff ref_labels is)

    Returns:
      A tuple containing:
        both_tot: the frequency of a particular bucket appearing in both output and reference
        ref_tot: the frequency of a particular bucket appearing in just reference
        out_tot: the frequency of a particular bucket appearing in just output
        rec: recall of the bucket
        prec: precision of the bucket
        fmeas: f1-measure of the bucket
    """
    if not hasattr(self, 'case_insensitive'):
      self.case_insensitive = False
      
    ref_labels = ref_labels if ref_labels else []
    out_labels = out_labels if out_labels else []
    matches = [[0, 0, 0] for x in self.bucket_strs]
    for ref_sent, out_sent, ref_lab, out_lab in itertools.zip_longest(ref, out, ref_labels, out_labels):
      ref_pos = defaultdict(lambda: [])
      for i, word in enumerate(ref_sent):
        if self.case_insensitive:
          word = corpus_utils.lower(word)
        ref_pos[word].append(i)
      for i, word in enumerate(out_sent):
        if self.case_insensitive:
          word = corpus_utils.lower(word)
        if len(ref_pos[word]) > 0:
          ri = ref_pos[word][0]
          ref_pos[word] = ref_pos[word][1:]
          bucket = self.calc_bucket(word,
                                    ref_label=ref_lab[ri] if ref_lab else None,
                                    out_label=out_lab[i] if out_lab else None)
          matches[bucket][0] += 1
          matches[bucket][1] += 1
        else:
          bucket = self.calc_bucket(word,
                                    out_label=out_lab[i] if out_lab else None)
        matches[bucket][2] += 1
      for word, my_pos in ref_pos.items():
        if len(my_pos) > 0:
          for ri in my_pos:
            bucket = self.calc_bucket(ref_sent[ri],
                                      ref_label=ref_lab[ri] if ref_lab else None)
            matches[bucket][1] += 1
    for both_tot, ref_tot, out_tot in matches:
      if both_tot == 0:
        rec, prec, fmeas = 0.0, 0.0, 0.0
      else:
        rec = both_tot / float(ref_tot)
        prec = both_tot / float(out_tot)
        fmeas = 2 * prec * rec / (prec + rec)
      yield both_tot, ref_tot, out_tot, rec, prec, fmeas

  def calc_source_bucketed_matches(self, src, ref, out, ref_aligns, out_aligns, src_labels=None):
    """
    Calculate the number of matches, bucketed by the type of word we have
    This must be used with a subclass that has self.bucket_strs defined, and self.calc_bucket(word) implemented.

    Args:
      src: The source corpus
      ref: The reference corpus
      out: The output corpus
      ref_aligns: Alignments of the reference corpus
      out_aligns: Alignments of the output corpus
      src_labels: Labels of the source corpus (optional)

    Returns:
      A tuple containing:
        both_tot: the frequency of a particular bucket appearing in both output and reference
        ref_tot: the frequency of a particular bucket appearing in just reference
        out_tot: the frequency of a particular bucket appearing in just output
        rec: recall of the bucket
        prec: precision of the bucket
        fmeas: f1-measure of the bucket
    """
    if not hasattr(self, 'case_insensitive'):
      self.case_insensitive = False

    src_labels = src_labels if src_labels else []
    matches = [[0, 0, 0] for x in self.bucket_strs]
    for src_sent, ref_sent, out_sent, ref_align, out_align, src_lab in itertools.zip_longest(src, ref, out, ref_aligns, out_aligns, src_labels):
      ref_cnt = defaultdict(lambda: 0)
      for i, word in enumerate(ref_sent):
        if self.case_insensitive:
          word = corpus_utils.lower(word)
        ref_cnt[word] += 1
      for i, align in enumerate(out_align):
        src_index, trg_index = align.split('-')
        src_index = int(src_index)
        trg_index = int(trg_index)
        src_word = src_sent[src_index]
        word = out_sent[trg_index]
        if self.case_insensitive:
          word = corpus_utils.lower(word)
        bucket = self.calc_bucket(src_word,
                                  src_label=src_lab[src_index] if src_lab else None)
        if ref_cnt[word] > 0:
          ref_cnt[word] -= 1
          matches[bucket][0] += 1
        matches[bucket][2] += 1
      for i, align in enumerate(ref_align):
        src_index, trg_index = align.split('-')
        src_index = int(src_index)
        trg_index = int(trg_index)
        src_word = src_sent[src_index]
        bucket = self.calc_bucket(src_word,
                                  src_label=src_lab[src_index] if src_lab else None)
        matches[bucket][1] += 1

    for both_tot, ref_tot, out_tot in matches:
      if both_tot == 0:
        rec, prec, fmeas = 0.0, 0.0, 0.0
      else:
        rec = both_tot / float(ref_tot)
        prec = both_tot / float(out_tot)
        fmeas = 2 * prec * rec / (prec + rec)
      yield both_tot, ref_tot, out_tot, rec, prec, fmeas

  def calc_bucketed_likelihoods(self, corpus, likelihoods):
    """
    Calculate the average of log likelihoods, bucketed by the type of word/label we have
    This must be used with a subclass that has self.bucket_strs defined, and self.calc_bucket(word) implemented.

    Args:
      corpus: The text/label corpus over which we compute the likelihoods
      likelihoods: The log-likelihoods corresponding to each word/label in the corpus

    Returns:
      the average log-likelihood bucketed by the type of word/label we have
    """
    if not hasattr(self, 'case_insensitive'):
      self.case_insensitive = False

    if type(corpus) == str:
      corpus = corpus_utils.load_tokens(corpus)
    bucketed_likelihoods = [[0.0, 0] for _ in self.bucket_strs]
    if len(corpus) != len(likelihoods):
      raise ValueError("Corpus and likelihoods should have the same size.")
    for sent, list_of_likelihoods in zip(corpus, likelihoods):
      if len(sent) != len(list_of_likelihoods):
        raise ValueError("Each sentence of the corpus should have likelihood value for each word")

      for word, ll in zip(sent, list_of_likelihoods):
        if self.case_insensitive:
          word = corpus_utils.lower(word)
        bucket = self.calc_bucket(word, ref_label=word)
        bucketed_likelihoods[bucket][0] += ll
        bucketed_likelihoods[bucket][1] += 1

    for ll, count in bucketed_likelihoods:
      if count != 0:
        yield ll/float(count)
      else:
        yield "NA" # not applicable


class FreqWordBucketer(WordBucketer):

  def __init__(self,
               freq_counts=None, freq_count_file=None, freq_corpus_file=None, freq_data=None,
               bucket_cutoffs=None,
               case_insensitive=False):
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
      case_insensitive: A boolean specifying whether to turn on the case insensitive option.
    """
    self.case_insensitive = case_insensitive
    if not freq_counts:
      freq_counts = defaultdict(lambda: 0)
      if freq_count_file != None:
        print(f'Reading frequency from "{freq_count_file}"')
        with open(freq_count_file, "r") as f:
          for line in f:
            word, freq = line.strip().split('\t')
            if self.case_insensitive:
              freq_counts[corpus_utils.lower(word)] = int(freq)
            else:
              freq_counts[word] = int(freq)
      elif freq_corpus_file:
        print(f'Reading frequency from "{freq_corpus_file}"')
        for words in corpus_utils.iterate_tokens(freq_corpus_file):
          for word in words:
            if self.case_insensitive:
              freq_counts[corpus_utils.lower(word)] += 1
            else:
              freq_counts[word] += 1
      elif freq_data:
        print('Reading frequency from the reference')
        for words in freq_data:
          for word in words:
            if self.case_insensitive:
              freq_counts[corpus_utils.lower(word)] += 1
            else:
              freq_counts[word] += 1
      else:
        raise ValueError('Must have at least one source of frequency counts for FreqWordBucketer')
    self.freq_counts = freq_counts

    if bucket_cutoffs is None:
      bucket_cutoffs = [1, 2, 3, 4, 5, 10, 100, 1000]
    self.set_bucket_cutoffs(bucket_cutoffs)

  def calc_bucket(self, word, ref_label=None, out_label=None, src_label=None):
    if self.case_insensitive:
      return self.cutoff_into_bucket(self.freq_counts.get(corpus_utils.lower(word), 0))
    else:
      return self.cutoff_into_bucket(self.freq_counts.get(word, 0))

  def name(self):
    return "frequency"

  def idstr(self):
    return "freq"

class LabelWordBucketer(WordBucketer):

  def __init__(self,
               label_set=None):
    """
    A bucketer that buckets words by their labels.

    Args:
      label_set: The set of labels to use as buckets. This can be a list, or a string separated by '+'s.
    """
    if type(label_set) == str:
      label_set = label_set.split('+')
    self.bucket_strs = label_set + ['other']
    label_set_len = len(label_set)
    self.bucket_map = defaultdict(lambda: label_set_len)
    for i, l in enumerate(label_set):
      self.bucket_map[l] = i

  def calc_bucket(self, word, ref_label=None, out_label=None, src_label=None):
    if ref_label:
      return self.bucket_map[ref_label]
    elif out_label:
      return self.bucket_map[out_label]
    elif src_label:
      return self.bucket_map[src_label]
    else:
      raise ValueError('When calculating buckets by label, ref_label or out_label must be non-zero')

  def name(self):
    return "labels"

  def idstr(self):
    return "labels"

class NumericalLabelWordBucketer(WordBucketer):

  def __init__(self,
               bucket_cutoffs=None):
    """
    A bucketer that buckets words by labels that are numerical values.

    Args:
      bucket_cutoffs: Cutoffs for each bucket.
                      The first bucket will be range(0,bucket_cutoffs[0]).
                      Middle buckets will be range(bucket_cutoffs[i],bucket_cutoffs[i-1].
                      Final bucket will be everything greater than bucket_cutoffs[-1].
    """
    if bucket_cutoffs is None:
      bucket_cutoffs = [0.25, 0.5, 0.75]
    self.set_bucket_cutoffs(bucket_cutoffs)

  def calc_bucket(self, word, ref_label=None, out_label=None, src_label=None):
    if ref_label:
      return self.cutoff_into_bucket(float(ref_label))
    elif out_label:
      return self.cutoff_into_bucket(float(out_label))
    elif src_label:
      return self.cutoff_into_bucket(float(src_label))
    else:
      raise ValueError('When calculating buckets by label, ref_label or out_label must be non-zero')

  def name(self):
    return "numerical labels"

  def idstr(self):
    return "numlabels"

class SentenceBucketer(Bucketer):

  def calc_bucket(self, val, ref=None, out_label=None, ref_label=None):
    """
    Calculate the bucket for a particular sentence

    Args:
      val: The sentence to calculate the bucket for
      ref: The reference sentence, if it exists
      ref_labels: The label of the reference sentence, if it exists
      out_labels: The label of the output sentence, if it exists

    Returns:
      An integer ID of the bucket
    """
    raise NotImplementedError('calc_bucket must be implemented in subclasses of SentenceBucketer')

  def create_bucketed_corpus(self, out, ref=None, ref_labels=None, out_labels=None):
    bucketed_corpus = [([],[] if ref else None) for _ in self.bucket_strs]
    if ref is None:
      ref = out

    if ref_labels is None:
      ref_labels = out_labels
    
    for i, (out_words, ref_words) in enumerate(zip(out, ref)):
      bucket = self.calc_bucket(out_words, ref=(ref_words if ref else None), label=(ref_labels[i][0] if ref_labels else None))
      bucketed_corpus[bucket][0].append(out_words)
      if ref != None:
        bucketed_corpus[bucket][1].append(ref_words)
    return bucketed_corpus

class ScoreSentenceBucketer(SentenceBucketer):
  """
  Bucket sentences by some score (e.g. BLEU)
  """

  def __init__(self, score_type, bucket_cutoffs=None, case_insensitive=False):
    self.score_type = score_type
    self.scorer = scorers.create_scorer_from_profile(score_type)
    if bucket_cutoffs is None:
      bucket_cutoffs = [x * self.scorer.scale / 10.0 for x in range(1,10)]
    self.set_bucket_cutoffs(bucket_cutoffs, num_type='float')
    self.case_insensitive = case_insensitive

  def calc_bucket(self, val, ref=None, label=None):
    if self.case_insensitive:
      return self.cutoff_into_bucket(self.scorer.score_sentence(corpus_utils.lower(ref), corpus_utils.lower(val))[0])
    else:
      return self.cutoff_into_bucket(self.scorer.score_sentence(ref, val)[0])

  def name(self):
    return self.scorer.name()

  def idstr(self):
    return self.scorer.idstr()

class LengthSentenceBucketer(SentenceBucketer):
  """
  Bucket sentences by length
  """

  def __init__(self, bucket_cutoffs=None):
    if bucket_cutoffs is None:
      bucket_cutoffs = [10, 20, 30, 40, 50, 60]
    self.set_bucket_cutoffs(bucket_cutoffs, num_type='int')

  def calc_bucket(self, val, ref=None, label=None):
    return self.cutoff_into_bucket(len(val))

  def name(self):
    return "length"

  def idstr(self):
    return "length"

class LengthDiffSentenceBucketer(SentenceBucketer):
  """
  Bucket sentences by length
  """

  def __init__(self, bucket_cutoffs=None):
    if bucket_cutoffs is None:
      bucket_cutoffs = [-20, -10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 11, 21]
    self.set_bucket_cutoffs(bucket_cutoffs, num_type='int')

  def calc_bucket(self, val, ref=None, label=None):
    return self.cutoff_into_bucket(len(val) - len(ref))

  def name(self):
    return "len(output)-len(reference)"

  def idstr(self):
    return "lengthdiff"

class LabelSentenceBucketer(SentenceBucketer):

  def __init__(self, label_set=None):
    """
    A bucketer that buckets sentences by their labels.

    Args:
      label_set: The set of labels to use as buckets. This can be a list, or a string separated by '+'s.
    """
    if type(label_set) == str:
      label_set = label_set.split('+')
    self.bucket_strs = label_set + ['other']
    label_set_len = len(label_set)
    self.bucket_map = defaultdict(lambda: label_set_len)
    for i, l in enumerate(label_set):
      self.bucket_map[l] = i

  def calc_bucket(self, val, ref=None, label=None):
    return self.bucket_map[label]

  def name(self):
    return "labels"

  def idstr(self):
    return "labels"

class NumericalLabelSentenceBucketer(SentenceBucketer):

  def __init__(self, bucket_cutoffs=None):
    """
    A bucketer that buckets sentences by labels that are numerical values.

    Args:
      bucket_cutoffs: Cutoffs for each bucket.
                      The first bucket will be range(0,bucket_cutoffs[0]).
                      Middle buckets will be range(bucket_cutoffs[i],bucket_cutoffs[i-1].
                      Final bucket will be everything greater than bucket_cutoffs[-1].
    """
    if bucket_cutoffs is None:
      bucket_cutoffs = [0.25, 0.5, 0.75]
    self.set_bucket_cutoffs(bucket_cutoffs)

  def calc_bucket(self, val, ref=None, label=None):
    return self.cutoff_into_bucket(float(label))

  def name(self):
    return "numerical labels"

  def idstr(self):
    return "numlabels"

def create_word_bucketer_from_profile(bucket_type,
                                      freq_counts=None, freq_count_file=None, freq_corpus_file=None, freq_data=None,
                                      label_set=None,
                                      bucket_cutoffs=None,
                                      case_insensitive=False):
  if type(bucket_cutoffs) == str:
    bucket_cutoffs = [arg_utils.parse_intfloat(x) for x in bucket_cutoffs.split(':')]
  if bucket_type == 'freq':
    return FreqWordBucketer(
      freq_counts=freq_counts,
      freq_count_file=freq_count_file,
      freq_corpus_file=freq_corpus_file,
      freq_data=freq_data,
      bucket_cutoffs=bucket_cutoffs,
      case_insensitive=case_insensitive)
  elif bucket_type == 'label':
    return LabelWordBucketer(
      label_set=label_set)
  elif bucket_type == 'numlabel':
    return NumericalLabelWordBucketer(
      bucket_cutoffs=bucket_cutoffs)
  else:
    raise ValueError(f'Illegal bucket type {bucket_type}')

def create_sentence_bucketer_from_profile(bucket_type,
                                          score_type=None,
                                          bucket_cutoffs=None,
                                          label_set=None,
                                          case_insensitive=False):
  if type(bucket_cutoffs) == str:
    bucket_cutoffs = [arg_utils.parse_intfloat(x) for x in bucket_cutoffs.split(':')]
  if bucket_type == 'score':
    return ScoreSentenceBucketer(score_type, bucket_cutoffs=bucket_cutoffs, case_insensitive=case_insensitive)
  elif bucket_type == 'length':
    return LengthSentenceBucketer(bucket_cutoffs=bucket_cutoffs)
  elif bucket_type == 'lengthdiff':
    return LengthDiffSentenceBucketer(bucket_cutoffs=bucket_cutoffs)
  elif bucket_type == 'label':
    return LabelSentenceBucketer(label_set=label_set)
  elif bucket_type == 'numlabel':
    return NumericalLabelSentenceBucketer(bucket_cutoffs=bucket_cutoffs)
  else:
    raise NotImplementedError(f'Illegal bucket type {bucket_type}')
