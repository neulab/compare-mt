import os.path
import unittest

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import scorers
from corpus_utils import load_tokens

class TestScorers(unittest.TestCase):

  def setUp(self):
    self.ref_sentence = ["By the end of this year , there 'll be nearly a billion people on this planet that actively use social networking sites ."]
    self.out_sentence = ["By the end of this year will be on this planet about billion people to use active aspects of social networks ."]

    example_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "example")
    self.ref_file = os.path.join(example_path, "ted.ref.eng")
    self.out_file = os.path.join(example_path, "ted.sys1.eng")

  def test_chrf_sentence(self):
    scorer = scorers.create_scorer_from_profile("chrf")
    chrf = scorer.score_sentence(self.ref_sentence, self.out_sentence)
    self.assertAlmostEqual(chrf, 0.59, places=2)
  
  def test_chrf_corpus(self):
    scorer = scorers.create_scorer_from_profile("chrf")
    ref = load_tokens(self.ref_file)
    out = load_tokens(self.out_file)
    chrf, _ = scorer.score_corpus(ref, out)
    self.assertAlmostEqual(chrf, 0.48, places=2)


if __name__ == '__main__':
  unittest.main()