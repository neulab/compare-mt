import nltk
import nltk.translate.chrf_score  # This is necessary to avoid an AttributeError in NLTK
import sacrebleu
import numpy as np
import math
import re
import subprocess
import tempfile
from collections import Counter
from itertools import chain

from compare_mt import corpus_utils
from compare_mt import align_utils
from compare_mt import ngram_utils
from compare_mt.rouge import rouge_scorer

# Global variable controlling scorer scale
global_scorer_scale = 100.0


class Scorer(object):

  @property
  def scale(self):
    return 1.0

  def score_corpus(self, ref, out, src=None):
    pass

  def score_sentence(self, ref, out, src=None):
    pass

  def cache_stats(self, ref, out, src=None):
    return None

  def name(self):
    """
    A name that can have spaces that describes the scorer.
    """
    return None

  def idstr(self):
    """
    An ID string that contains no spaces but identifies the scorer.
    """
    return None

class SentenceFactoredScorer(Scorer):
  def score_corpus(self, ref, out, src=None):
    """
    Score a corpus using the average of the score

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source corpus. Might be ignored or required 
        depending on the metric
    Returns:
      A tuple containing a single value for the average score, and None
    """
    if len(ref) == 0:
      return 0.0, None
    score_sum = 0
    src = [None for _ in ref] if src is None else src
    for r, o, s in zip(ref, out, src):
      score_sum += self.score_sentence(r, o, s)[0]
    return score_sum/len(ref), None

  def cache_stats(self, ref, out, src=None):
    """
    Cache sufficient statistics for caculating scores

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source corpus. Might be ignored or required 
        depending on the metric
    Returns:
      A tuple of cached statistics
    """
    if hasattr(self, 'case_insensitive') and self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    cached_scores = []
    src = [None for _ in ref] if src is None else src
    for r, o, s in zip(ref, out, src):
      cached_scores.append(self.score_sentence(r, o, s)[0])

    return cached_scores

  def score_cached_corpus(self, sent_ids, cached_stats):
    """
    Score a corpus with cache

    Args:
      sent_ids: The sentence ids for reference and output corpora
      cached_stats: A tuple of cached statistics

    Returns:
      A tuple containing a single value for the score and a string summarizing auxiliary information
    """
    cached_stats = np.array(cached_stats)
    return np.mean(cached_stats[sent_ids]), None

class BleuScorer(Scorer):
  """
  A scorer that calculates BLEU score.
  """
  def __init__(self, weights=(0.25, 0.25, 0.25, 0.25), case_insensitive=False):
    self.weights = weights
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def score_corpus(self, ref, out, src=None):
    """
    Score a corpus using BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed

    Returns:
      A tuple containing a single value for the BLEU score and a string summarizing auxiliary information
    """
    cached_stats = self.cache_stats(ref, out)
    return self.score_cached_corpus(range(len(ref)), cached_stats)

  def score_sentence(self, ref, out, src=None):
    raise NotImplementedError("Sentence-level calculation is not implemented in BleuScorer as it is usually 0."
                              "Consider using SentenceBleuScorer (string sentbleu) instead.")

  def _precision(self, ref, out, n):
    """
    Caculate n-gram precision 

    Args:
      ref: A reference sentence
      out: An output sentence

    Returns:
      Numerator and denominator of the precision
    """
    out_ngram = ngram_utils.sent_ngrams_list(out, n)
    ref_ngram = ngram_utils.sent_ngrams_list(ref, n)
    out_cnt = Counter(out_ngram)
    ref_cnt = Counter(ref_ngram)

    num = 0
    denom = 0
    for ngram, o_cnt in out_cnt.items():
      num += min(o_cnt, ref_cnt[ngram])
      denom += o_cnt
    denom = max(1, denom)

    return num, denom

  def cache_stats(self, ref, out, src=None):
    """
    Cache sufficient statistics for caculating BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed

    Returns:
      A list of cached statistics
    """
    if self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    cached_stats = []

    for r, o in zip(ref, out):
      prec = []
      for n in range(1, len(self.weights) + 1):
        prec.append(self._precision(r, o, n))
      cached_stats.append( (len(r), len(o), prec) )

    return cached_stats

  def score_cached_corpus(self, sent_ids, cached_stats):
    """
    Score a corpus using BLEU score with cache

    Args:
      sent_ids: The sentence ids for reference and output corpora
      cached_stats: A list of cached statistics

    Returns:
      A tuple containing a single value for the BLEU score and a string summarizing auxiliary information
    """
    if len(cached_stats) == 0:
      return 0.0, None

    cached_ref_len, cached_out_len, cached_prec = zip(*cached_stats)

    num_prec = Counter()
    denom_prec = Counter()

    ref_len = 0
    out_len = 0
    for sent_id in sent_ids:
      ref_len += cached_ref_len[sent_id]
      out_len += cached_out_len[sent_id]
      for n in range(1, len(self.weights) + 1):
        num, denom = cached_prec[sent_id][n-1]
        num_prec[n] += num
        denom_prec[n] += denom

    if num_prec[1] == 0:
      return 0, None

    prec = 0
    for i, w in enumerate(self.weights, start=1):
      p = num_prec[i] / denom_prec[i] if denom_prec[i] != 0 else 0
      p = math.log(p) if p > 0 else 0
      prec += p * w

    bp = min(1, math.exp(1 - ref_len/out_len)) if out_len != 0 else 0

    return self.scale * bp * math.exp(prec), None

  def name(self):
    return "BLEU"

  def idstr(self):
    return "bleu"

class SentBleuScorer(SentenceFactoredScorer):
  """
  A scorer that calculates sentence-level smoothed BLEU score.
  """
  def __init__(self, case_insensitive=False):
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def score_sentence(self, ref, out, src=None):
    """
    Score a single sentence with sentence-level smoothed BLEU score

    Args:
      ref: A reference sentence
      out: An output sentence
      src: A source sentence. Ignored if passed

    Returns:
      The sentence-level BLEU score, and None
    """
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    if self.case_insensitive:
      bleu_score = nltk.translate.bleu_score.sentence_bleu([corpus_utils.lower(ref)], corpus_utils.lower(out), smoothing_function=chencherry.method2)
    else:
      bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], out, smoothing_function=chencherry.method2)
    return self.scale * bleu_score, None

  def name(self):
    return "sentence-level BLEU"

  def idstr(self):
    return "sentbleu"

class LengthScorer(Scorer):
  """
  A scorer that calculate the length ratio
  """
  def score_corpus(self, ref, out, src=None):
    """
    Calculate the length ratio for a corpus

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed

    Returns:
      A tuple containing a single value for the length ratio and a string summarizing auxiliary information
    """
    ref_words = sum([len(x) for x in ref])
    out_words = sum([len(x) for x in out])
    if ref_words == 0:
      return 0.0, f'ref={ref_words}, out={out_words}'
    return self.scale * out_words / ref_words, f'ref={ref_words}, out={out_words}'

  def score_sentence(self, ref, out, src=None):
    """
    Score a single sentence by length ratio

    Args:
      ref: A reference sentence
      out: An output sentence
      src: A source sentence. Ignored if passed

    Returns:
      The length, and a string summarizing the length of the reference and output sentence
    """
    if len(ref) == 0:
      return 0.0, f"ref={len(ref)}, out={len(out)}"
    return len(out) / len(ref), f"ref={len(ref)}, out={len(out)}"

  def name(self):
    return "length ratio"

  def idstr(self):
    return "lengthrat"

class ExactMatchScorer(Scorer):
  """
  A scorer that calculates exact matches
  """
  def score_corpus(self, ref, out, src=None):
    """
    Calculate the percentage of exact matches in a corpus

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed

    Returns:
      A tuple containing a single value for the exact match percentage and None
    """
    matches = 0
    for r, o in zip(ref, out):
      if r == o:
        matches += 1
    return float(matches) / len(ref), None

  def score_sentence(self, ref, out, src=None):
    """
    Score a single sentence by exact match

    Args:
      ref: A reference sentence
      out: An output sentence
      src: A source sentence. Ignored if passed

    Returns:
      1 if exact matches 0, and None
    """
    return 1.0 if ref == out else 0, None

  def name(self):
    return "exact match"

  def idstr(self):
    return "exact"

class RibesScorer(SentenceFactoredScorer):
  """
  A scorer that calculates RIBES score.
  """
  def __init__(self, order=-1, alpha=0.25, beta=0.1, case_insensitive=False):
    self.order = order
    self.alpha = alpha
    self.beta = beta
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def _kendall_tau_distance(self, alignment):
    """
    Caculate the Kendall's tau distance for RIBES

    Args:
      alignment: an alignment represented as a list of integers

    Returns:
      The Kendall's tau distance
    """
    dis = 0
    n = len(alignment)
    if n <= 1:
      return 0
    for i in range(n):
      for j in range(i+1, n):
        if alignment[j] > alignment[i]:
          dis += 1
    return 2*dis/(n*n-n)

  def score_sentence(self, ref, out, src=None):
    """
    Score a single sentence with RIBES score

    Args:
      ref: A reference sentence
      out: An output sentence
      src: A source sentence. Ignored if passed

    Returns:
      The RIBES score, and None
    """
    alignment = align_utils.ngram_context_align(ref, out, order=self.order, case_insensitive=self.case_insensitive)
    kt_dis = self._kendall_tau_distance(alignment)
    prec = len(alignment)/ len(out) if len(out) != 0 else 0
    bp = min(1, math.exp(1-len(ref)/len(out))) if len(out) != 0 else 0
    return self.scale * kt_dis * (prec**self.alpha) * (bp**self.beta), None

  def name(self):
    return "RIBES"

  def idstr(self):
    return "ribes"


class SacreBleuScorer(Scorer):
  """
  A scorer that computes BLEU on detokenized text.

  """
  def __init__(self, smooth_method='exp', smooth_value=0, use_effective_order=False, case_insensitive=False):
    self.smooth_method = smooth_method
    self.smooth_value = smooth_value
    self.use_effective_order = use_effective_order
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def score_sentence(self, ref, out, src=None):
    raise NotImplementedError("Sentence-level calculation is not implemented in SacreBleuScorer as it is usually 0."
                              "Consider using SentenceBleuScorer (string sentbleu) instead.")

  def score_corpus(self, ref, out, src=None):
    cached_stats = self.cache_stats(ref, out)
    return self.score_cached_corpus(range(len(ref)), cached_stats)

  def cache_stats(self, ref, out, src=None):
    """
    Cache sufficient statistics for caculating SacreBLEU score

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed

    Returns:
      A list of cached statistics
    """
    if self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    cached_stats = []
    for r, o in zip(ref, out):
      re = sacrebleu.corpus_bleu(" ".join(o), " ".join(r))
      cached_stats.append( (re.counts, re.totals, re.sys_len, re.ref_len) )

    return cached_stats

  def score_cached_corpus(self, sent_ids, cached_stats):
    """
    Score a corpus using SacreBLEU score with cache

    Args:
      sent_ids: The sentence ids for reference and output corpora
      cached_stats: A list of cached statistics

    Returns:
      A tuple containing a single value for the SacreBLEU score and a string summarizing auxiliary information
    """
    if len(cached_stats) == 0:
      return 0.0, None

    counts, totals, sys_len, ref_len = zip(*cached_stats)
    counts, totals, sys_len, ref_len = [np.sum(np.array(x)[sent_ids], 0) for x in [counts, totals, sys_len, ref_len]]

    return sacrebleu.compute_bleu(counts, totals, sys_len, ref_len, smooth_method=self.smooth_method, smooth_value=self.smooth_value, use_effective_order=self.use_effective_order).score, None

  def name(self):
    return "SacreBleuScorer"

  def idstr(self):
    return "sacrebleu"


class ChrFScorer(Scorer):
  """
  A scorer that calculates chrF (character n-gram F-score) score.

  This computes F2 score (beta=2.0 as per http://www.aclweb.org/anthology/W16-2341).
  """
  def __init__(self, case_insensitive=False):
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def chrf_score(self, refs, out):
    return self.scale * nltk.translate.chrf_score.corpus_chrf(
      [[" ".join(x) for x in ref] for ref in refs],
      [" ".join(x) for x in out],
      max_len=6,  # Order 6 n-grams
      beta=2.0,  # F2 score
      ignore_whitespace=True  # No whitespaces
    )

  def score_corpus(self, ref, out, src=None):
    """
    Score a corpus using ChrF score

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed

    Returns:
      A tuple containing a single value for the ChrF score and a string summarizing auxiliary information
    """
    if self.case_insensitive:
      chrf = self.chrf_score([[corpus_utils.lower(x)] for x in ref], corpus_utils.lower(out))
    else:
      chrf = self.chrf_score([[x] for x in ref], out)
    return chrf, None

  def score_sentence(self, ref, out, src=None):
    return self.chrf_score([ref], [out]), None

  def name(self):
    return "ChrF"

  def idstr(self):
    return "chrf"

class RougeScorer(SentenceFactoredScorer):
  """
  A scorer that calculates ROUGE score.
  """
  def __init__(self, rouge_type, score_type='fmeasure', use_stemmer=False, case_insensitive=False):
    self.rouge_type = rouge_type
    self.score_type = score_type
    self._stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def score_sentence(self, ref, out, src=None):
    if self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    if self._stemmer:
      ref = [self._stemmer.stem(x) if len(x) > 3 else x for x in ref]
      out = [self._stemmer.stem(x) if len(x) > 3 else x for x in out]

    if self.rouge_type == 'rougeL':
      ref, out = self.tokenize(" ".join(ref)), self.tokenize(" ".join(out))
      scores = rouge_scorer._score_lcs(ref, out)
    elif self.rouge_type == 'rougeLsum':
      refs = [self.tokenize(s) for s in self.get_sents(ref)]
      outs = [self.tokenize(s) for s in self.get_sents(out)]
      scores = rouge_scorer._summary_level_lcs(refs, outs)
    elif re.match(r"rouge[0-9]$", self.rouge_type):
      ref, out = self.tokenize(" ".join(ref)), self.tokenize(" ".join(out))
      n = int(self.rouge_type[5:])
      if n <= 0:
        raise ValueError(f"rougen requires positive n: {self.rouge_type}")
      ref_ngrams = rouge_scorer._create_ngrams(ref, n)
      out_ngrams = rouge_scorer._create_ngrams(out, n)
      scores = rouge_scorer._score_ngrams(ref_ngrams, out_ngrams)
    else:
      raise ValueError(f"Invalid rouge type: {self.rouge_type}")


    if self.score_type == 'fmeasure':
      score_value = scores.fmeasure
    elif self.score_type == 'precision':
      score_value = scores.precision
    elif self.score_type == 'recall':
      score_value = scores.recall
    else:
      raise ValueError(f"Invalid score type: {self.score_type}")

    return self.scale * score_value, None

  def get_sents(self, tokens):
    # assume sentences are separated by "."
    sents = " ".join(tokens).split(".")
    sents = [x for x in sents if len(x)]
    return sents

  def tokenize(self, tokens):
    text = re.sub(r"[^a-zA-Z0-9]+", " ", tokens)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if len(x)]
    return tokens

  def name(self):
    return self.rouge_type

  def idstr(self):
    return self.rouge_type.lower()

class WERScorer(Scorer):
  """
  A scorer that calculates Word Error Rate (WER).
  """
  def __init__(self, sub_pen=1.0, ins_pen=1.0, del_pen=1.0, case_insensitive=False):
    self.sub_pen = 1.0
    self.ins_pen = 1.0
    self.del_pen = 1.0
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def score_corpus(self, ref, out, src=None):
    """
    Score a corpus using WER

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed

    Returns:
      A tuple containing a single value for the WER and None
    """
    cached_stats = self.cache_stats(ref, out)
    return self.score_cached_corpus(np.arange(len(ref)), cached_stats)

  def score_sentence(self, ref, out, src=None):
    return self.score_corpus([ref], [out])

  def cache_stats(self, ref, out, src=None):
    """
    Cache sufficient statistics for caculating WER

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A list of cached statistics
    """
    cached_stats = []

    for r, o in zip(ref, out):
      cached_stats.append( (len(r), self._edit_distance(r, o)) )

    return cached_stats

  def score_cached_corpus(self, sent_ids, cached_stats):
    """
    Score a corpus with cache

    Args:
      sent_ids: The sentence ids for reference and output corpora
      cached_stats: A list of cached statistics

    Returns:
      A tuple containing a single value for the score and a string summarizing auxiliary information
    """
    if len(cached_stats) == 0:
      return 0.0, None

    cached_ref_len, cached_edit_distance = zip(*cached_stats)
    cached_ref_len, cached_edit_distance = np.array(cached_ref_len), np.array(cached_edit_distance)
    denom = np.sum(cached_ref_len[sent_ids])
    wer = np.sum(cached_edit_distance[sent_ids])/denom if denom != 0 else 0
    return self.scale * wer, None

  def _edit_distance(self, ref, out, src=None):
    if self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    sp1 = len(ref)+1
    tp1 = len(out)+1
    scores = np.zeros((sp1, tp1))
    equals = (np.expand_dims(np.array(ref), axis=1) == np.array(out))
    scores[:,0] = range(sp1)
    scores[0,:] = range(tp1)

    # Forward edit distance
    for i in range(0, len(ref)):
      for j in range(0, len(out)):
        my_action = 0 if equals[i,j] else 1
        my_score = scores[i,j] + my_action * self.sub_pen
        del_score = scores[i,j+1] + self.del_pen
        if del_score < my_score:
          my_score = del_score
        ins_score = scores[i+1,j] + self.ins_pen
        if ins_score < my_score:
          my_score = ins_score
        scores[i+1,j+1] = my_score

    return scores[-1,-1]

  def name(self):
    return "Word Error Rate"

  def idstr(self):
    return "wer"

class METEORScorer(Scorer):
  """
  A scorer that calculates METEOR score.
  """
  def __init__(self, meteor_directory, options=None):
    self.meteor_directory = meteor_directory
    self.options = options
    self.weights, self.parameters = self._get_weights_and_parameters(options)

  @property
  def scale(self):
    return global_scorer_scale

  def score_corpus(self, ref, out, src=None):
    """
    Score a corpus using METEOR score
    Args:
      ref: A reference corpus
      out: An output corpus
    Returns:
      A tuple containing a single value for the METEOR score and a string summarizing auxiliary information
    """
    cached_stats = self.cache_stats(ref, out)
    return self.score_cached_corpus(np.arange(len(ref)), cached_stats)

  def score_sentence(self, ref, out):
    return self.score_corpus([ref], [out])

  def cache_stats(self, ref, out, src=None):
    """
    Cache sufficient statistics for caculating METEOR score
    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source courpus. Ignored if passed
    Returns:
      A list of cached statistics
    """
    with tempfile.TemporaryDirectory() as directory:
      ref_name = directory + '/ref'
      out_name = directory + '/out'

      corpus_utils.write_tokens(ref_name, ref)
      corpus_utils.write_tokens(out_name, out)

      cached_stats = []

      command = f'java -Xmx2G -jar {self.meteor_directory}/meteor-*.jar {out_name} {ref_name} '
      if self.options:
        command += self.options
      command += ' -ssOut'

      p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
      stats = p.communicate()[0].decode("utf-8").split('\n')[:-1]

      for stat_str in stats:
        stat = tuple(float(x) for x in stat_str.split())
        cached_stats.append(stat)

    return cached_stats

  def score_cached_corpus(self, sent_ids, cached_stats):
    """
    Score a corpus using METEOR score with cache
    Args:
      sent_ids: The sentence ids for reference and output corpora
      cached_stats: A list of cached statistics
    Returns:
      A tuple containing a single value for the METEOR score and a string summarizing auxiliary information
    """
    if len(cached_stats) == 0:
      return 0.0, None

    cached_stats = np.array(cached_stats)

    # compute sufficient statistics
    sent_stats = cached_stats[sent_ids]

    # num_total_chunks = sum(num_sent_chunks) - minus_chunk
    minus_chunk = 0
    for stat in sent_stats:
      out_len = stat[0]
      ref_len = stat[1]
      out_total_match = stat[4] + stat[6] + stat[8] + stat[10] + stat[12] + stat[14] + stat[16] + stat[18]
      ref_total_match = stat[5] + stat[7] + stat[9] + stat[11] + stat[13] + stat[15] + stat[17] + stat[19]
      if out_len == out_total_match and ref_len == ref_total_match and stat[-3] == 1:
        minus_chunk += 1

    cal_stats = np.sum(sent_stats, 0)
    cal_stats[20] -= minus_chunk

    # rename
    alpha, beta, gamma, delta = self.parameters
    out_len, ref_len = cal_stats[0], cal_stats[1]
    out_func_words, ref_func_words = cal_stats[2], cal_stats[3]
    out_content_match_stage = np.array([cal_stats[4], cal_stats[8], cal_stats[12], cal_stats[16]])
    ref_content_match_stage = np.array([cal_stats[5], cal_stats[9], cal_stats[13], cal_stats[17]])
    out_func_match_stage = np.array([cal_stats[6], cal_stats[10], cal_stats[14], cal_stats[18]])
    ref_func_match_stage = np.array([cal_stats[7], cal_stats[11], cal_stats[15], cal_stats[19]])
    chunks = cal_stats[20]
    out_word_match, ref_word_match = cal_stats[21], cal_stats[22]

    # compute the METEOR score
    out_weighted_len = delta * (out_len-out_func_words) + (1.0-delta) * out_func_words
    ref_weighted_len = delta * (ref_len-ref_func_words) + (1.0-delta) * ref_func_words

    out_weighted_match = np.sum(self.weights * (out_content_match_stage*delta + out_func_match_stage*(1-delta)))
    ref_weighted_match = np.sum(self.weights * (ref_content_match_stage*delta + ref_func_match_stage*(1-delta)))

    prec = out_weighted_match / out_weighted_len if out_weighted_len != 0 else 0
    recall = ref_weighted_match / ref_weighted_len if ref_weighted_len != 0 else 0
    fmean = 1.0 / ( (1.0-alpha)/prec + alpha/recall ) if prec != 0 and recall != 0 else 0

    out_total_match = np.sum(out_content_match_stage) + np.sum(out_func_match_stage)
    ref_total_match = np.sum(ref_content_match_stage) + np.sum(ref_func_match_stage)

    frag = float(chunks) / (float(out_word_match+ref_word_match)/2)
    frag = 0 if out_total_match == out_len and ref_total_match == ref_len and chunks == 1 else frag

    frag_penalty = gamma * math.pow(frag, beta)

    score = fmean * (1.0-frag_penalty)

    return self.scale * score, None

  def _get_weights_and_parameters(self, options):
    if self.options is None:
      return (np.array([1.0, 0.6, 0.8, 0.6]), np.array([0.85, 0.2, 0.6, 0.75]))

    weights, parameters = np.zeros(4), np.zeros(4)
    # a simple and (maybe) slow way to obtain weights and parameters
    with tempfile.TemporaryDirectory() as directory:
      ref_name = directory + '/ref'
      out_name = directory + '/out'

      corpus_utils.write_tokens(ref_name, [["test"]])
      corpus_utils.write_tokens(out_name, [["test"]])

      command = f'java -Xmx2G -jar {self.meteor_directory}/meteor-*.jar {out_name} {ref_name} {options}'

      p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
      stats = p.communicate()[0].decode("utf-8").split()

      weights_index = stats.index('Weights:') + 1
      params_index = stats.index('Parameters:') + 1
      for i in range(4):
        weights[i] = float(stats[weights_index+i])
        parameters[i] = float(stats[params_index+i])

    return weights, parameters

  def name(self):
    return "METEOR"

  def idstr(self):
    return "meteor"

class COMETScorer(SentenceFactoredScorer):
  """
  A scorer that calculates sentence-level COMET score.
  """
  def __init__(self, model_name="wmt-large-da-estimator-1719"):
    import torch
    from comet.models import download_model
    self.cuda = torch.cuda.is_available()
    self.model = download_model(model_name)

  @property
  def scale(self):
    return global_scorer_scale

  def score_sentence(self, ref, out, src=None):
    """
    Score a single sentence with sentence-level COMET score

    Args:
      ref: A reference sentence
      out: An output sentence
      src: A source sentence

    Returns:
      The sentence-level COMET  score, and None
    """
    assert src is not None, "COMET requires source"

    data = [
      {"src": " ".join(src), "mt": " ".join(out), "ref": " ".join(ref)}
    ]
    score = self.model.predict(data, cuda=self.cuda)[1][0]
    return self.scale * score, None

  def name(self):
    return "sentence-level COMET"

  def idstr(self):
    return "comet"

class GleuScorer(Scorer):
  """
  A scorer that calculates GLEU score.

  References:
    "Ground Truth for Grammatical Error Correction Metrics", Napoles et al.
    "GLEU Without Tuning", Napoles et al.
  """
  def __init__(self, weights=(0.25, 0.25, 0.25, 0.25), case_insensitive=False):
    self.weights = weights
    self.case_insensitive = case_insensitive

  @property
  def scale(self):
    return global_scorer_scale

  def score_corpus(self, ref, out, src=None):
    """
    Score a corpus using GLEU score

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source corpus. Required

    Returns:
      A tuple containing a single value for the GLEU score and a string summarizing auxiliary information
    """
    cached_stats = self.cache_stats(ref, out, src)
    return self.score_cached_corpus(range(len(ref)), cached_stats)

  def score_sentence(self, ref, out, src=None):
    """
    Score a sentence using GLEU score

    Args:
      ref: A reference sentence
      out: An output sentence
      src: A source sentence. Required

    Returns:
      A tuple containing a single value for the GLEU score and a string summarizing auxiliary information
    """
    cached_stats = self.cache_stats([ref], [out], [src])
    # Smooth according to https://github.com/cnap/gec-ranking/blob/master/scripts/gleu.py
    stat = cached_stats[0]
    cached_stats[0] = (stat[0], stat[1],
                       [(max(num, 1), max(denom, 1)) for num, denom in stat[2]])
    return self.score_cached_corpus(range(1), cached_stats)

  def _precision(self, ref, out, src, n):
    """
    Calcualte GLEU-specific n-gram precision

    Args:
      ref: A reference sentence
      out: An output sentence
      src: A source sentence

    Returns:
      Numerator and denominator of the precision
    """
    ref_ngram = ngram_utils.sent_ngrams_list(ref, n)
    out_ngram = ngram_utils.sent_ngrams_list(out, n)
    src_ngram = ngram_utils.sent_ngrams_list(src, n)
    ref_cnt = Counter(ref_ngram)
    out_cnt = Counter(out_ngram)
    src_cnt = Counter(src_ngram)

    out_join_ref = out_cnt & ref_cnt
    out_join_src = out_cnt & src_cnt

    num = sum(out_join_ref.values()) - \
          sum((out_join_src - out_join_ref).values())
    # According to https://github.com/cnap/gec-ranking/blob/master/scripts/gleu.py
    num = max(num, 0)
    denom = sum(out_cnt.values())

    return num, denom

  def cache_stats(self, ref, out, src=None):
    """
    Cache sufficient statistics for calculating BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus
      src: A source corpus. Required.

    Returns:
      A list of cached statistics
    """
    if self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)
      src = corpus_utils.lower(src)

    cached_stats = []
    for r, o, s in zip(ref, out, src):
      prec = []
      for n in range(1, len(self.weights) + 1):
        prec.append(self._precision(r, o, s, n))
      cached_stats.append((len(r), len(o), prec))
    return cached_stats

  def score_cached_corpus(self, sent_ids, cached_stats):
    """
    Score a corpus using GLEU score with cache

    Args:
      sent_ids: The sentence ids for reference and output corpora
      cached_stats: A list of cached statistics

    Returns:
      A tuple containing a single value for the GLEU score and a string summarizing auxiliary information
    """
    if len(cached_stats) == 0:
      return 0.0, None

    cached_ref_len, cached_out_len, cached_prec = zip(*cached_stats)

    num_prec = Counter()
    denom_prec = Counter()

    ref_len = 0
    out_len = 0
    for sent_id in sent_ids:
      ref_len += cached_ref_len[sent_id]
      out_len += cached_out_len[sent_id]
      for n in range(1, len(self.weights) + 1):
        num, denom = cached_prec[sent_id][n-1]
        num_prec[n] += num
        denom_prec[n] += denom

    # According to https://github.com/cnap/gec-ranking/blob/master/scripts/gleu.py
    if any(map(lambda x: x == 0, chain(num_prec.values(), denom_prec.values()))):
      return 0, None

    prec = 0
    for i, w in enumerate(self.weights, start=1):
      p = math.log(num_prec[i] / denom_prec[i])
      prec += p * w

    bp = min(1, math.exp(1 - ref_len/out_len)) if out_len != 0 else 0

    return self.scale * bp * math.exp(prec), None

  def name(self):
    return "GLEU"

  def idstr(self):
    return "gleu"

def create_scorer_from_profile(profile, case_insensitive=False, meteor_directory=None, options=None):
  """
  Create a scorer from a profile string
  Args:
    profile: a profile string of "bleu" for BLEU or "length" for length ratio
    case_insensitive: A boolean specifying whether to turn on the case insensitive option

  Returns:
    A scorer to perform the appropriate scoring
  """
  if profile == 'bleu':
    return BleuScorer(case_insensitive=case_insensitive)
  if profile == 'sacrebleu':
    return SacreBleuScorer(case_insensitive=case_insensitive)
  elif profile == 'sentbleu':
    return SentBleuScorer(case_insensitive=case_insensitive)
  elif profile == 'length':
    return LengthScorer()
  elif profile == 'ribes':
    return RibesScorer(case_insensitive=case_insensitive)
  elif profile == 'chrf':
    return ChrFScorer(case_insensitive=case_insensitive)
  elif re.match(r"rouge[0-9L](sum)?$", profile):
    return RougeScorer(rouge_type=profile, case_insensitive=case_insensitive)
  elif profile == 'wer':
    return WERScorer(case_insensitive=case_insensitive)
  elif profile == 'meteor':
    if meteor_directory == None:
      raise ValueError("Must specify the directory of the METEOR source code.")
    return METEORScorer(meteor_directory=meteor_directory, options=options)
  elif profile == 'exact':
    return ExactMatchScorer()
  elif profile == 'comet':
    return COMETScorer()
  elif profile == 'gleu':
    return GleuScorer()
  else:
    raise ValueError(f'Invalid profile for scorer {profile}'.format(profile=profile))
