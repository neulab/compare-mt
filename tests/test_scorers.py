import os.path
import unittest
import numpy as np
import sys

compare_mt_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(compare_mt_root)

from compare_mt import scorers
from compare_mt.corpus_utils import load_tokens


def _get_example_data():
  example_path = os.path.join(compare_mt_root, "example")
  ref_file = os.path.join(example_path, "ted.ref.eng")
  out1_file = os.path.join(example_path, "ted.sys1.eng")
  out2_file = os.path.join(example_path, "ted.sys2.eng")
  return [load_tokens(x) for x in (ref_file, out1_file, out2_file)]

def _get_example_data_detokenized():
  example_path = os.path.join(compare_mt_root, "example")
  ref_file = os.path.join(example_path, "ted.ref.detok.eng")
  out1_file = os.path.join(example_path, "ted.sys1.detok.eng")
  out2_file = os.path.join(example_path, "ted.sys2.detok.eng")
  return [load_tokens(x) for x in (ref_file, out1_file, out2_file)]


class TestBleuScorer(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out1, self.out2 = _get_example_data()
    self.ids = list(range(len(self.ref)))
    self.scorer = scorers.create_scorer_from_profile("bleu", case_insensitive=False)
    self.cache_stats1 = self.scorer.cache_stats(self.ref, self.out1)
    self.cache_stats2 = self.scorer.cache_stats(self.ref, self.out2)
    self.n_random_retries = 10

  def test_score_corpus(self):
    bleu, _ = self.scorer.score_corpus(self.ref, self.out1)
    # Compare to moses multi-bleu.pl
    self.assertAlmostEqual(bleu, 22.44, places=1)
  
  def test_score_sentence(self):
    
    def should_raise():
      return self.scorer.score_sentence(self.ref[0], self.out1[0])
    
    self.assertRaises(NotImplementedError, should_raise)


  def test_score_cached_corpus(self):
    for _ in range(self.n_random_retries):
      np.random.shuffle(self.ids)
      random_ids = self.ids[:int(len(self.ids)*0.5)]

      # compare-mt implementation
      my_sys1_score, _ = self.scorer.score_cached_corpus(random_ids, self.cache_stats1)
      my_sys2_score, _ = self.scorer.score_cached_corpus(random_ids, self.cache_stats2)

      # nltk implementation
      random_ref = [self.ref[i] for i in random_ids]
      random_sys1 = [self.out1[i] for i in random_ids]
      random_sys2 = [self.out2[i] for i in random_ids]
      nltk_sys1_score, _ = self.scorer.score_corpus(random_ref, random_sys1)
      nltk_sys2_score, _ = self.scorer.score_corpus(random_ref, random_sys2)

      self.assertAlmostEqual(my_sys1_score, nltk_sys1_score)
      self.assertAlmostEqual(my_sys2_score, nltk_sys2_score)


class TestSentBleuScorer(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out, _ = _get_example_data()
    self.scorer = scorers.create_scorer_from_profile("sentbleu")

  def test_score_sentence(self):
    bleu, _ = self.scorer.score_sentence(self.ref[0], self.out[0])
    # compare to nltk
    self.assertAlmostEqual(bleu, 32.607099228782377)
  
  def test_score_corpus(self):
    sent_bleu_corpus, _ = self.scorer.score_corpus(self.ref, self.out)
    avg_sent_bleu = sum([self.scorer.score_sentence(ref_sent, out_sent)[0]
                         for ref_sent, out_sent in zip(self.ref, self.out)])
    avg_sent_bleu /= len(self.ref)
    # compare to sacrebleu --force --metrics=chrf
    self.assertAlmostEqual(sent_bleu_corpus, avg_sent_bleu)


class TestLengthScorer(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out, _ = _get_example_data()
    self.scorer = scorers.create_scorer_from_profile("length")

  def test_score_sentence(self):
    length_ratio, desc = self.scorer.score_sentence(self.ref[0], self.out[0])
    self.assertAlmostEqual(length_ratio, 22 / 24)
    self.assertEqual(desc, "ref=24, out=22")
  
  def test_score_corpus(self):
    length_ratio_corpus, desc = self.scorer.score_corpus(self.ref, self.out)
    self.assertAlmostEqual(length_ratio_corpus, 45672 / 48183)
    self.assertEqual(desc, "ref=48183, out=45672")



class TestRibesScorer(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out, _ = _get_example_data()
    self.scorer = scorers.create_scorer_from_profile("ribes")

  def test_score_sentence(self):
    ribes, _ = self.scorer.score_sentence(self.ref[0], self.out[0])
    self.assertAlmostEqual(ribes, 84.9014, 4)
  
  def test_score_corpus(self):
    ribes_corpus, _ = self.scorer.score_corpus(self.ref, self.out)
    self.assertAlmostEqual(ribes_corpus, 80.0020, 4)


class TestChrFScorer(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out, _ = _get_example_data()
    self.scorer = scorers.create_scorer_from_profile("chrf")

  def test_chrf_sentence(self):
    chrf, _ = self.scorer.score_sentence(self.ref[0], self.out[0])
    # compare to sacrebleu --force --metrics=chrf
    self.assertAlmostEqual(chrf, 59, places=0)
  
  def test_chrf_corpus(self):
    chrf, _ = self.scorer.score_corpus(self.ref, self.out)
    # compare to sacrebleu --force --metrics=chrf
    self.assertAlmostEqual(chrf, 48, places=0)


class TestSacreBleuScorer(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out, _ = _get_example_data_detokenized()
    self.scorer = scorers.create_scorer_from_profile("sacrebleu")

  def test_detok_bleu_corpus(self):
    detok_bleu, _ = self.scorer.score_corpus(self.ref, self.out)
    # compare to sacrebleu
    self.assertAlmostEqual(detok_bleu, 21.7, places=0)


class TestGleuScorer(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    example_path = os.path.join(compare_mt_root, "example")
    filenames = ["ted.ref.eng", "ted.sys1.eng", "ted.orig.slk"]
    cls.ref, cls.out, cls.src = [load_tokens(os.path.join(example_path, name)) for name in filenames]
    cls.scorer = scorers.create_scorer_from_profile("gleu", case_insensitive=False)

  def test_score_corpus(self):
    gleu, _ = self.scorer.score_corpus(self.ref, self.out, self.src)
    # Compare to https://github.com/cnap/gec-ranking
    self.assertAlmostEqual(gleu, 22.39, places=1)

  def test_score_sentence(self):
    src = "A simple src sentence of test .".split()
    ref = "A simple source sentence for testing .".split()
    out = "A simple src sentence for testing .".split()
    gleu, _ = self.scorer.score_sentence(ref, out, src)
    # Compare to https://github.com/cnap/gec-ranking
    self.assertAlmostEqual(gleu, 33.03, places=1)

  def test_score_cached_corpus(self):
    cached_stats = [
      (9, 2, [(2, 2), (1, 1), (0, 0), (0, 0)]),
      (4, 13, [(4, 13), (2, 12), (0, 11), (0, 10)]),
      (10, 10, [(6, 10), (4, 9), (1, 8), (0, 7)])
    ]
    gleu, _ = self.scorer.score_cached_corpus(range(len(cached_stats)), cached_stats)
    self.assertEqual(gleu, 0)


if __name__ == "__main__":
  unittest.main()
