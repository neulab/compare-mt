import os.path
import unittest

import sys
compare_mt_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(compare_mt_root)
import scorers
import numpy as np
from corpus_utils import load_tokens

import nltk.translate.ribes_score

def _get_example_data():
  example_path = os.path.join(compare_mt_root, "example")
  ref_file = os.path.join(example_path, "ted.ref.eng")
  out1_file = os.path.join(example_path, "ted.sys1.eng")
  out2_file = os.path.join(example_path, "ted.sys2.eng")
  return [load_tokens(x) for x in (ref_file, out1_file, out2_file)]

ref, out, out2 = _get_example_data()
scorer = scorers.create_scorer_from_profile("ribes")
print(f'Ours first sentence: {scorer.score_sentence(ref[0], out[0])[0]}')
ribes_corpus, _ = scorer.score_corpus(ref, out)
print(f'Ours corpus: {ribes_corpus}')
print(f'NLTK first sentence: {nltk.translate.ribes_score.sentence_ribes([ref[0]], out[0])}')
try:
  nltk_ribes = nltk.translate.ribes_score.corpus_ribes([ref], out)
  print(f'NLTK corpus: {nltk_ribes}')
except ZeroDivisionError:
  print('NLTK corpus: ZeroDivisionError')


