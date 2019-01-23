import nltk
import nltk.translate.chrf_score  # This is necessary to avoid an AttributeError in NLTK
import math
from collections import Counter
import corpus_utils
import align_utils
import ngram_utils

class Scorer(object):
  def score_corpus(self, ref, out):
    pass
  
  def score_sentence(self, ref, out):
    pass

  def cache_stats(self, ref, out):
    return None

  def name(self):
    return None

class SentenceFactoredScorer(Scorer):
  def cache_stats(self, ref, out):
    """
    Cache sufficient statistics for caculating scores

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple of cached statistics
    """
    if hasattr(self, 'case_insensitive') and self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    cached_scores = []
    for r, o in zip(ref, out):
      cached_scores.append(self.score_sentence(r, o)[0])
  
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
    import numpy as np
    cached_stats = np.array(cached_stats)
    return np.mean(cached_stats[sent_ids]), None
    

class BleuScorer(Scorer):
  """
  A scorer that calculates BLEU score.
  """
  def __init__(self, weights=(0.25, 0.25, 0.25, 0.25), case_insensitive=False):
    self.weights = weights
    self.case_insensitive = case_insensitive

  def score_corpus(self, ref, out):
    """
    Score a corpus using BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the BLEU score and a string summarizing auxiliary information
    """
    cached_stats = self.cache_stats(ref, out)
    return self.score_cached_corpus(range(len(ref)), cached_stats)

  def score_sentence(self, ref, out):
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
  
  def cache_stats(self, ref, out):
    """
    Cache sufficient statistics for caculating BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple of cached statistics
    """
    if self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    cached_ref_len = []
    cached_out_len = []
    cached_prec = []

    for r, o in zip(ref, out):
      cached_ref_len.append(len(r))
      cached_out_len.append(len(o))
      prec = []
      for n in range(1, len(self.weights) + 1):
        prec.append(self._precision(r, o, n))
      cached_prec.append(prec)

    return (cached_ref_len, cached_out_len, cached_prec)

  def score_cached_corpus(self, sent_ids, cached_stats):
    """
    Score a corpus using BLEU score with cache

    Args:
      sent_ids: The sentence ids for reference and output corpora
      cached_stats: A tuple of cached statistics

    Returns:
      A tuple containing a single value for the BLEU score and a string summarizing auxiliary information
    """
    cached_ref_len, cached_out_len, cached_prec = cached_stats

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
      return 0

    prec = 0
    for i, w in enumerate(self.weights, start=1):
      p = num_prec[i] / denom_prec[i] if denom_prec[i] != 0 else 0
      p = math.log(p) if p > 0 else 0
      prec += p * w 
    
    bp = min(1, math.exp(1 - ref_len/out_len)) if out_len != 0 else 0

    return bp * math.exp(prec), None

  def name(self):
    return "BLEU"

class SentBleuScorer(SentenceFactoredScorer):
  """
  A scorer that calculates sentence-level smoothed BLEU score.
  """
  def __init__(self, case_insensitive=False):
    self.case_insensitive = case_insensitive

  def score_corpus(self, ref, out):
    """
    Score a corpus using the average of sentence-level BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the average sentence BLEU, and None
    """
    bleu_sum = 0
    for r, o in zip(ref, out):
      bleu_sum += self.score_sentence(r, o)[0]
    return bleu_sum/len(ref), None

  def score_sentence(self, ref, out):
    """
    Score a single sentence with sentence-level smoothed BLEU score

    Args:
      ref: A reference sentence
      out: An output sentence

    Returns:
      The sentence-level BLEU score, and None
    """
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    if self.case_insensitive:
      return nltk.translate.bleu_score.sentence_bleu([corpus_utils.lower(ref)], corpus_utils.lower(out), smoothing_function=chencherry.method2), None
    else:  
      return nltk.translate.bleu_score.sentence_bleu([ref], out, smoothing_function=chencherry.method2), None

  def name(self):
    return "sentence-level BLEU"

class LengthScorer(Scorer):
  """
  A scorer that calculate the length ratio
  """
  def score_corpus(self, ref, out):
    """
    Calculate the length ratio for a corpus

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the length ratio and a string summarizing auxiliary information
    """
    ref_words = sum([len(x) for x in ref])
    out_words = sum([len(x) for x in out])
    return out_words/ref_words, f'ref={ref_words}, out={out_words}'

  def score_sentence(self, ref, out):
    """
    Score a single sentence by length ratio

    Args:
      ref: A reference sentence
      out: An output sentence

    Returns:
      The length, and a string summarizing the length of the reference and output sentence
    """
    return len(out) / len(ref), f"ref={len(ref)}, out={len(out)}"

  def name(self):
    return "length ratio"

class RibesScorer(SentenceFactoredScorer):
  """
  A scorer that calculates RIBES score.
  """
  def __init__(self, order=2, alpha=0.25, beta=0.1, case_insensitive=False):
    self.order = order
    self.alpha = alpha
    self.beta = beta
    self.case_insensitive = case_insensitive

  def score_corpus(self, ref, out):
    """
    Score a corpus using the average of RIBES score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the average sentence RIBES, and None
    """
    ribes_sum = 0
    for r, o in zip(ref, out):
      ribes_sum += self.score_sentence(r, o)[0]
    return ribes_sum/len(ref), None

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

  def score_sentence(self, ref, out):
    """
    Score a single sentence with RIBES score

    Args:
      ref: A reference sentence
      out: An output sentence

    Returns:
      The RIBES score, and None
    """
    alignment = align_utils.ngram_context_align(ref, out, order=self.order, case_insensitive=self.case_insensitive)
    kt_dis = self._kendall_tau_distance(alignment) 
    prec = len(alignment)/ len(out) if len(out) != 0 else 0
    bp = min(1, math.exp(1-len(ref)/len(out))) if len(out) != 0 else 0
    return kt_dis * (prec**self.alpha) * (bp**self.beta), None

  def name(self):
    return "RIBES"


class ChrFScorer(Scorer):
  """
  A scorer that calculates chrF (character n-gram F-score) score.

  This computes F2 score (beta=2.0 as per http://www.aclweb.org/anthology/W16-2341).
  """
  def __init__(self, case_insensitive=False):
    self.case_insensitive = case_insensitive

  def chrf_score(self, refs, out):
    return nltk.translate.chrf_score.corpus_chrf(
      [[" ".join(x) for x in ref] for ref in refs],
      [" ".join(x) for x in out],
      max_len=6,  # Order 6 n-grams
      beta=2.0,  # F2 score
      ignore_whitespace=True  # No whitespaces
    )

  def score_corpus(self, ref, out):
    """
    Score a corpus using ChrF score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the ChrF score and a string summarizing auxiliary information
    """
    if self.case_insensitive:
      chrf = self.chrf_score([[corpus_utils.lower(x)] for x in ref], corpus_utils.lower(out))
    else:
      chrf = self.chrf_score([[x] for x in ref], out)
    return chrf, None

  def score_sentence(self, ref, out):
    return self.chrf_score([ref], [out]), None

  def name(self):
    return "ChrF"


def create_scorer_from_profile(profile, case_insensitive=False):
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
  elif profile == 'sentbleu':
    return SentBleuScorer(case_insensitive=case_insensitive)
  elif profile == 'length':
    return LengthScorer()
  elif profile == 'ribes':
    return RibesScorer(case_insensitive=case_insensitive)
  elif profile == 'chrf':
    return ChrFScorer(case_insensitive=case_insensitive)
  else:
    raise ValueError(f'Invalid profile for scorer {profile}')
