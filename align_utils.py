import itertools
import numpy
from collections import defaultdict

def alignment_to_permutation(aligns, src_len):
  '''
  Convert alignments to permutations.
  '''
  src_trg = {}
  for src_pos, trg_pos in zip(aligns['src'], aligns['trg']):
    src_trg[src_pos] = min(src_trg[src_pos], trg_pos) if src_pos in src_trg else trg_pos

  pos_value = []
  for i in range(src_len):
    if i in src_trg:
      v = src_trg[i]
    else:
      def last_v(pos):
        if pos == -1:
          return 0
        return src_trg[pos] if pos in src_trg else last_v(pos - 1)
      v = last_v(i - 1) 
    pos_value.append(v)
  return numpy.argsort(pos_value)

def kendall_tau_distance(perm_1, perm_2):
  '''
  Compute Kendall's tau distance between two permutations.
  The distance metric range from 1 (a perfect match) to 0 (maximum disagreement).
  '''
  dis = 0
  n = len(perm_1)
  if n == 1:
    return 1
  for i in range(n):
    for j in range(n):
      if perm_1[i] < perm_1[j] and perm_2[i] > perm_2[j]:
        dis += 1
  return 1 - numpy.sqrt(2*dis/(n*n-n))
      
def load_alignment(line):
  src_list = []
  trg_list = []
  for align in line:
    src_pos, trg_pos = align.split('-')
    src_list.append(int(src_pos))
    trg_list.append(int(trg_pos))
  return {'src':src_list, 'trg':trg_list}
     
def compute_kendall_tau_distance(ref_align, out1_align, out2_align, src_len=None):
  score_1 = 0
  score_2 = 0
  mono_score_1 = 0
  mono_score_2 = 0
  for i, [ref_a, out1_a, out2_a] in enumerate(itertools.zip_longest(ref_align, out1_align, out2_align)):
    ref_a, out1_a, out2_a = [load_alignment(x) for x in (ref_a, out1_a, out2_a)]
    len_i = src_len[i] if src_len else max([max(x['src'])+1 for x in (ref_a, out1_a, out2_a)])
    ref_p, out1_p, out2_p = [alignment_to_permutation(x, len_i) for x in (ref_a, out1_a, out2_a)]
    score_1 += kendall_tau_distance(ref_p, out1_p)
    score_2 += kendall_tau_distance(ref_p, out2_p)
    mono_score_1 += kendall_tau_distance(numpy.arange(len_i), out1_p)
    mono_score_2 += kendall_tau_distance(numpy.arange(len_i), out2_p)
  return score_1/(i+1), score_2/(i+1), mono_score_1/(i+1), mono_score_2/(i+1)
