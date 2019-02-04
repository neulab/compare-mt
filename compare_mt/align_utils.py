from collections import defaultdict
from compare_mt import corpus_utils

def _count_ngram(sent, order):
  gram_pos = dict()
  for i in range(order):
    gram_pos[i+1] = defaultdict(lambda: [])
  for i, word in enumerate(sent):
    for j in range(min(i+1, order)):
      gram_pos[j+1][word].append(i-j)
      word = sent[i-j-1] + ' ' + word
  return gram_pos

def ngram_context_align(ref, out, order=-1, case_insensitive=False):
  """
  Calculate the word alignment between a reference sentence and an output sentence. 
  Proposed in the following paper:

  Automatic Evaluation of Translation Quality for Distant Language Pairs
  Hideki Isozaki, Tsutomu Hirao, Kevin Duh, Katsuhito Sudoh, Hajime Tsukada
  http://www.anthology.aclweb.org/D/D10/D10-1092.pdf 

  Args:
    ref: A reference sentence
    out: An output sentence
    order: The highest order of grams we want to consider (-1=inf)
    case_insensitive: A boolean specifying whether to turn on the case insensitive option

  Returns:
    The word alignment, represented as a list of integers. 
  """

  if case_insensitive:
    ref = corpus_utils.lower(ref)
    out = corpus_utils.lower(out)

  order = len(ref) if order == -1 else order

  ref_gram_pos = _count_ngram(ref, order)
  out_gram_pos = _count_ngram(out, order)

  worder = []
  for i, word in enumerate(out):
    if len(ref_gram_pos[1][word]) == 0:
      continue
    if len(ref_gram_pos[1][word]) == len(out_gram_pos[1][word]) == 1:
      worder.append(ref_gram_pos[1][word][0])
    else:
      word_forward = word 
      word_backward = word 
      for j in range(1, order):
        if i - j >= 0:
          word_backward = out[i-j] + ' ' + word_backward 
          if len(ref_gram_pos[j+1][word_backward]) == len(out_gram_pos[j+1][word_backward]) == 1:
            worder.append(ref_gram_pos[j+1][word_backward][0]+j)
            break

        if i + j < len(out):
          word_forward = word_forward + ' ' + out[i+j]
          if len(ref_gram_pos[j+1][word_forward]) == len(out_gram_pos[j+1][word_forward]) == 1:
            worder.append(ref_gram_pos[j+1][word_forward][0])
            break

  return worder
