import nltk
import nltk.translate.chrf_score  # This is necessary to avoid an AttributeError in NLTK
import math
import corpus_utils
import align_utils

class BleuScorer:
  """
  A scorer that calculates BLEU score.
  """
  def __init__(self, case_insensitive=False):
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
    if self.case_insensitive:
      bleu = nltk.translate.bleu_score.corpus_bleu([[corpus_utils.lower(x)] for x in ref], corpus_utils.lower(out))
    else:
      bleu = nltk.translate.bleu_score.corpus_bleu([[x] for x in ref], out)
    return bleu, None

  def score_sentence(self, ref, out):
    raise NotImplementedError("Sentence-level calculation is not implemented in BleuScorer as it is usually 0."
                              "Consider using SentenceBleuScorer (string sentbleu) instead.")

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
    bp = min(1, math.exp(1-len(ref)/len(out)))
    return kt_dis * (prec**self.alpha) * (bp**self.beta), None

  def name(self):
    return "RIBES"


class ChrFScorer:
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
    return self.chrf_score([ref], [out])

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
