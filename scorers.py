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

  def score_sentence(self, ref, out):
    raise NotImplementedError("Sentence-level calculation is not implemented in BleuScorer as it is usually 0."
                              "Consider using SentenceBleuScorer (string sentbleu) instead.")

  def name(self):
    return "BLEU"

class SentBleuScorer:
  """
  A scorer that calculates sentence-level smoothed BLEU score.
  """
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
    return nltk.translate.bleu_score.sentence_bleu([ref], out, smoothing_function=chencherry.method2), None

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
    return "length ratio"

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
  elif profile == 'sentbleu':
    return SentBleuScorer()
  elif profile == 'length':
    return LengthScorer()
  else:
    raise ValueError(f'Invalid profile for scorer {profile}')