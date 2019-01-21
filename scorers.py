import nltk
import math
from collections import Counter, defaultdict
import corpus_utils
import align_utils
import ngram_utils

class BleuScorer:
  """
  A scorer that calculates BLEU score.
  """
  def __init__(self, case_insensitive=False):
    self.case_insensitive = case_insensitive
    self.cached_ref_len = {}
    self.cached_out_len = {}
    self.cached_prec = {}

  def score_corpus(self, ref, out):
    """
    Score a corpus using BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the BLEU score and a string summarizing auxiliary information
    """
    if self.case_insensitive:
      bleu = nltk.translate.bleu_score.corpus_bleu([[corpus_utils.lower(x)] for x in ref], corpus_utils.lower(out))
    else:
      bleu = nltk.translate.bleu_score.corpus_bleu([[x] for x in ref], out)
    return bleu, None

  def score_sentence(self, ref, out):
    raise NotImplementedError("Sentence-level calculation is not implemented in BleuScorer as it is usually 0."
                              "Consider using SentenceBleuScorer (string sentbleu) instead.")

  def cache_stats(self, cache_id, ref, out, weights=(0.25, 0.25, 0.25, 0.25)):
    def precision(ref, out, n):
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
  
    self.cached_ref_len[cache_id] = defaultdict(lambda: 0)
    self.cached_out_len[cache_id] = defaultdict(lambda: 0)
    self.cached_prec[cache_id] = defaultdict(lambda: {})
    for sent_id, (r, o) in enumerate(zip(ref, out)):
      self.cached_ref_len[cache_id][sent_id] = len(r)
      self.cached_out_len[cache_id][sent_id] = len(o)
      for n in range(1, len(weights) + 1):
        self.cached_prec[cache_id][sent_id][n] = precision(r, o, n)

  def fast_score_corpus(self, cache_id, sent_ids, weights=(0.25, 0.25, 0.25, 0.25)):
    if cache_id not in self.cached_ref_len:
      raise ValueError("Must cache first.")

    num_prec = Counter()
    denom_prec = Counter()
  
    ref_len = 0
    out_len = 0
    for sent_id in sent_ids:
      ref_len += self.cached_ref_len[cache_id][sent_id]
      out_len += self.cached_out_len[cache_id][sent_id]
      for n in range(1, len(weights) + 1):
        num, denom = self.cached_prec[cache_id][sent_id][n]
        num_prec[n] += num
        denom_prec[n] += denom

    if num_prec[1] == 0:
      return 0

    prec = 0
    for i, w in enumerate(weights, start=1):
      p = num_prec[i] / denom_prec[i] if denom_prec[i] != 0 else 0
      p = math.log(p) if p > 0 else 0
      prec += p * w 
    
    bp = min(1, math.exp(1 - ref_len/out_len)) if out_len != 0 else 0

    return bp * math.exp(prec)

  def name(self):
    return "BLEU"

class SentBleuScorer:
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

class LengthScorer:
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

  def name(self):
    return "length ratio"

class RibesScorer:
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

  def score_sentence(self, ref, out):
    """
    Score a single sentence with RIBES score

    Args:
      ref: A reference sentence
      out: An output sentence

    Returns:
      The RIBES score, and None
    """
    def kendall_tau_distance(alignment):
      dis = 0
      n = len(alignment)
      if n <= 1:
        return 0
      for i in range(n):
        for j in range(i+1, n):
          if alignment[j] > alignment[i]:
            dis += 1
      return 2*dis/(n*n-n)  
    alignment = align_utils.ngram_context_align(ref, out, order=self.order, case_insensitive=self.case_insensitive)
    kt_dis = kendall_tau_distance(alignment) 
    prec = len(alignment)/ len(out)
    bp = min(1, math.exp(1-len(ref)/len(out))) if len(out) != 0 else 0
    return kt_dis * (prec**self.alpha) * (bp**self.beta), None

  def name(self):
    return "RIBES"

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
  else:
    raise ValueError(f'Invalid profile for scorer {profile}')
