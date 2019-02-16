import nltk
import nltk.translate.chrf_score  # This is necessary to avoid an AttributeError in NLTK
import numpy as np
import math
import re
import subprocess
import tempfile
from collections import Counter
from compare_mt import corpus_utils
from compare_mt import align_utils
from compare_mt import ngram_utils
from compare_mt.rouge import rouge_scorer

class Scorer(object):
  def score_corpus(self, ref, out):
    pass
  
  def score_sentence(self, ref, out):
    pass

  def cache_stats(self, ref, out):
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
  def score_corpus(self, ref, out):
    """
    Score a corpus using the average of the score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the average score, and None
    """
    if len(ref) == 0:
      return 0.0, None
    score_sum = 0
    for r, o in zip(ref, out):
      score_sum += self.score_sentence(r, o)[0]
    return score_sum/len(ref), None

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

    return bp * math.exp(prec), None

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

  def idstr(self):
    return "sentbleu"

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
    if ref_words == 0:
      return 0.0, f'ref={ref_words}, out={out_words}'
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
    if len(ref) == 0:
      return 0.0, f"ref={len(ref)}, out={len(out)}"
    return len(out) / len(ref), f"ref={len(ref)}, out={len(out)}"

  def name(self):
    return "length ratio"

  def idstr(self):
    return "lengthrat"

class RibesScorer(SentenceFactoredScorer):
  """
  A scorer that calculates RIBES score.
  """
  def __init__(self, order=-1, alpha=0.25, beta=0.1, case_insensitive=False):
    self.order = order
    self.alpha = alpha
    self.beta = beta
    self.case_insensitive = case_insensitive

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

  def idstr(self):
    return "ribes"


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
  
  def score_sentence(self, ref, out):
    if self.case_insensitive:
      ref = corpus_utils.lower(ref)
      out = corpus_utils.lower(out)

    if self._stemmer:
      ref = [self._stemmer.stem(x) if len(x) > 3 else x for x in ref]
      out = [self._stemmer.stem(x) if len(x) > 3 else x for x in out]
    
    if self.rouge_type == 'rougeL':
      scores = rouge_scorer._score_lcs(ref, out)
    elif re.match(r"rouge[0-9]$", self.rouge_type):
      n = int(self.rouge_type[5:])
      if n <= 0:
        raise ValueError(f"rougen requires positive n: {self.rouge_type}")
      ref_ngrams = rouge_scorer._create_ngrams(ref, n)
      out_ngrams = rouge_scorer._create_ngrams(out, n)
      scores = rouge_scorer._score_ngrams(ref_ngrams, out_ngrams)
    else:
      raise ValueError(f"Invalid rouge type: {self.rouge_type}")

    if self.score_type == 'fmeasure':
      return scores.fmeasure, None
    elif self.score_type == 'precision':
      return scores.precision, None
    elif self.score_type == 'recall':
      return scores.recall, None
    else:
      raise ValueError(f"Invalid score type: {self.score_type}")

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

  def score_corpus(self, ref, out):
    """
    Score a corpus using WER

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the WER and None
    """
    cached_stats = self.cache_stats(ref, out)
    return self.score_cached_corpus(np.arange(len(ref)), cached_stats)

  def score_sentence(self, ref, out):
    return self.score_corpus([ref], [out])

  def cache_stats(self, ref, out):
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
    return wer, None

  def _edit_distance(self, ref, out):
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

  def score_corpus(self, ref, out):
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

  def cache_stats(self, ref, out):
    """
    Cache sufficient statistics for caculating METEOR score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A list of cached statistics
    """
    with tempfile.TemporaryDirectory() as directory:
      ref_name = directory + '/ref'
      out_name = directory + '/out'

      with open(ref_name, 'w') as f:
        corpus_utils.write_tokens(f, ref)
      with open(out_name, 'w') as f:
        corpus_utils.write_tokens(f, out)
      

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
    cal_stats = cached_stats[sent_ids]
    cal_stats = np.sum(cal_stats, 0)
    str_stats = corpus_utils.list2str(cal_stats)

    command = f'echo "EVAL ||| {str_stats}" | java -Xmx2G -jar {self.meteor_directory}/meteor-*.jar - - '
    if self.options:
      command += self.options
    command += ' -stdio'

    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    score = float(p.communicate()[0].decode("utf-8"))

    return score, None

  def name(self):
    return "METEOR"

  def idstr(self):
    return "meteor"

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
  elif profile == 'sentbleu':
    return SentBleuScorer(case_insensitive=case_insensitive)
  elif profile == 'length':
    return LengthScorer()
  elif profile == 'ribes':
    return RibesScorer(case_insensitive=case_insensitive)
  elif profile == 'chrf':
    return ChrFScorer(case_insensitive=case_insensitive)
  elif re.match(r"rouge[0-9L]$", profile):
    return RougeScorer(rouge_type=profile, case_insensitive=case_insensitive)
  elif profile == 'wer':
    return WERScorer(case_insensitive=case_insensitive)
  elif profile == 'meteor':
    if meteor_directory == None:
      raise ValueError("Must specify the directory of the METEOR source code.")
    return METEORScorer(meteor_directory=meteor_directory, options=options)
  else:
    raise ValueError(f'Invalid profile for scorer {profile}')
