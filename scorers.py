import nltk

class BleuScorer:
  """
  A scorer that calculates BLEU score.
  """
  def score_corpus(self, ref, out):
    """
    Score a corpus using BLEU score

    Args:
      ref: A reference corpus
      out: An output corpus

    Returns:
      A tuple containing a single value for the BLEU score and a string summarizing auxiliary information
    """
    bleu = nltk.translate.bleu_score.corpus_bleu([[x] for x in ref], out)
    return bleu, None

  def name(self):
    return "BLEU"

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
    return "Length Ratio"

def create_scorer_from_profile(profile):
  """
  Create a scorer from a profile string
  Args:
    profile: a profile string of "bleu" for BLEU or "length" for length ratio

  Returns:
    A scorer to perform the appropriate scoring
  """
  if profile == 'bleu':
    return BleuScorer()
  elif profile == 'length':
    return LengthScorer()
  else:
    raise ValueError(f'Invalid profile for scorer {profile}')