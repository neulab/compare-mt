import itertools
from collections import defaultdict
import math

def count_ngram(sent, order=2):
  gram_pos = dict()
  for i in range(order):
    gram_pos[i+1] = defaultdict(lambda: [])

  for i, word in enumerate(sent):
    for j in range(min(i+1, order)):
      gram_pos[j+1][word].append(i-j)
      word = sent[i-j-1] + ' ' + word
  
  return gram_pos

def word_alignment(ref, out, order=2):
  ref_gram_pos = count_ngram(ref, order)
  out_gram_pos = count_ngram(out, order)

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
        if i + j < len(out):
          word_forward = word_forward + ' ' + out[i+j]
          if len(ref_gram_pos[j+1][word_forward]) == len(out_gram_pos[j+1][word_forward]) == 1:
            worder.append(ref_gram_pos[j+1][word_forward][0])
            break

        if i - j >= 0:
          word_backward = out[i-j] + ' ' + word_backward 
          if len(ref_gram_pos[j+1][word_backward]) == len(out_gram_pos[j+1][word_backward]) == 1:
            worder.append(ref_gram_pos[j+1][word_backward][0]+j)
            break

  return worder

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

def RIBES_score(ref, out, order=2, alpha=0.25, beta=0.1):
  alignment = word_alignment(ref, out, order)
  kt_dis = kendall_tau_distance(alignment)
  prec = len(alignment) / len(out)
  bp = min(1, math.exp(1-len(ref)/len(out)))
  return kt_dis * (prec**alpha) * (bp**beta)
       
def compute_RIBES(ref, out1, out2, order=2, alpha=0.25, beta=0.1):
  score_1 = 0
  score_2 = 0
  for i, [r, o1, o2] in enumerate(itertools.zip_longest(ref, out1, out2)):
    score_1 += RIBES_score(r, o1, order, alpha, beta)
    score_2 += RIBES_score(r, o2, order, alpha, beta)
  return score_1/(i+1), score_2/(i+1)

