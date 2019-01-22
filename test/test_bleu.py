from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import unittest
import numpy as np
import corpus_utils
import scorers

class TestBLEU(unittest.TestCase):  
  def tearDown(self):
    pass

  def setUp(self):
    pass

  @classmethod
  def tearDownClass(self):
    pass

  @classmethod
  def setUpClass(self):
    ref_file = 'example/ted.ref.eng'
    out1_file = 'example/ted.sys1.eng'
    out2_file = 'example/ted.sys2.eng'
    self.ref, self.out1, self.out2 = [corpus_utils.load_tokens(x) for x in (ref_file, out1_file, out2_file)]
    self.ids = list(range(len(self.ref)))
    self.scorer = scorers.create_scorer_from_profile("bleu", case_insensitive=False)
    self.cache_stats1 = self.scorer.cache_stats(self.ref, self.out1)
    self.cache_stats2 = self.scorer.cache_stats(self.ref, self.out2)

  def test_run(self):
    for _ in range(100):
      np.random.shuffle(self.ids)
      random_ids = self.ids[:int(len(self.ids)*0.5)]

      # my implementation
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
        
        
if __name__ == '__main__':
  suite = unittest.TestSuite()
  suite.addTest(TestBLEU('test_run'))
  unittest.TextTestRunner().run(suite)
