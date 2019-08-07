import os.path
import unittest
import numpy as np
import sys

compare_mt_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(compare_mt_root)

from compare_mt import scorers
from compare_mt.corpus_utils import load_tokens, load_alignments
from compare_mt import compare_mt_main
from compare_mt import reporters

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


class TestScoreCache(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out1, self.out2 = _get_example_data()

  def test_score_cache(self):
    cached_stats1 = compare_mt_main.generate_score_report(self.ref, [self.out1], to_cache=True)
    cached_stats2 = compare_mt_main.generate_score_report(self.ref, [self.out2], to_cache=True)
    self.assertTrue('scores' in cached_stats1 and 'strs' in cached_stats1 and 'sign_stats' in cached_stats1)
    self.assertTrue('scores' in cached_stats2 and 'strs' in cached_stats2 and 'sign_stats' in cached_stats2)
    self.assertAlmostEqual(cached_stats1['scores'], 22.44, places=1)
    reporters.sys_names = [f'sys{i+1}' for i in range(2)]
    cached_report = compare_mt_main.generate_score_report(self.ref, [self.out1, self.out2], cache_dicts=[cached_stats1, cached_stats2], title='Aggregate Scores')
    ori_report = compare_mt_main.generate_score_report(self.ref, [self.out1, self.out2], title='Aggregate Scores')
    self.assertTrue(cached_report.scores == ori_report.scores)
    self.assertTrue(cached_report.strs == ori_report.strs)
    self.assertTrue(cached_report.wins == ori_report.wins)

  def test_score_cache_bootstrap(self):
    cached_stats1 = compare_mt_main.generate_score_report(self.ref, [self.out1], to_cache=True)
    cached_stats2 = compare_mt_main.generate_score_report(self.ref, [self.out2], to_cache=True)
    self.assertTrue('scores' in cached_stats1 and 'strs' in cached_stats1 and 'sign_stats' in cached_stats1)
    self.assertTrue('scores' in cached_stats2 and 'strs' in cached_stats2 and 'sign_stats' in cached_stats2)
    self.assertAlmostEqual(cached_stats1['scores'], 22.44, places=1)
    reporters.sys_names = [f'sys{i+1}' for i in range(2)]
    cached_report = compare_mt_main.generate_score_report(self.ref, [self.out1, self.out2], cache_dicts=[cached_stats1, cached_stats2], bootstrap=5, title='Aggregate Scores')
    ori_report = compare_mt_main.generate_score_report(self.ref, [self.out1, self.out2], bootstrap=5, title='Aggregate Scores')
    self.assertTrue(cached_report.scores == ori_report.scores)
    self.assertTrue(cached_report.strs == ori_report.strs)
    self.assertTrue(cached_report.wins == ori_report.wins)
    
class TestWordAccCache(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out1, self.out2 = _get_example_data()
 
  def test_wordacc_cache(self):
    cached_stats1 = compare_mt_main.generate_word_accuracy_report(self.ref, [self.out1], to_cache=True)
    cached_stats2 = compare_mt_main.generate_word_accuracy_report(self.ref, [self.out2], to_cache=True)
    self.assertTrue('statistics' in cached_stats1 and 'my_ref_total_list' in cached_stats1 and 'my_out_matches_list' in cached_stats1)
    self.assertTrue('statistics' in cached_stats2 and 'my_ref_total_list' in cached_stats2 and 'my_out_matches_list' in cached_stats2)
    ori_report = compare_mt_main.generate_word_accuracy_report(self.ref, [self.out1, self.out2])
    cached_report = compare_mt_main.generate_word_accuracy_report(self.ref, [self.out1, self.out2], cache_dicts=[cached_stats1, cached_stats2])
    self.assertTrue(cached_report.statistics == ori_report.statistics)
    self.assertTrue(cached_report.examples == ori_report.examples)

class TestSrcWordAccCache(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    example_path = os.path.join(compare_mt_root, "example")
    self.ref, self.out1, self.out2 = _get_example_data()
    src_file = os.path.join(example_path, "ted.orig.slk")
    self.src = load_tokens(src_file)
    self.ref_align_file = os.path.join(example_path, "ted.ref.align")
 
  def test_src_wordacc_cache(self):
    cached_stats1 = compare_mt_main.generate_src_word_accuracy_report(self.ref, [self.out1], self.src, ref_align_file=self.ref_align_file, to_cache=True)
    cached_stats2 = compare_mt_main.generate_src_word_accuracy_report(self.ref, [self.out2], self.src, ref_align_file=self.ref_align_file, to_cache=True)
    self.assertTrue('statistics' in cached_stats1 and 'my_ref_total_list' in cached_stats1 and 'my_out_matches_list' in cached_stats1)
    self.assertTrue('statistics' in cached_stats2 and 'my_ref_total_list' in cached_stats2 and 'my_out_matches_list' in cached_stats2)
    ori_report = compare_mt_main.generate_src_word_accuracy_report(self.ref, [self.out1, self.out2], self.src, ref_align_file=self.ref_align_file)
    cached_report = compare_mt_main.generate_src_word_accuracy_report(self.ref, [self.out1, self.out2], self.src, ref_align_file=self.ref_align_file, cache_dicts=[cached_stats1, cached_stats2])
    self.assertTrue(cached_report.statistics == ori_report.statistics)
    self.assertTrue(cached_report.examples == ori_report.examples)

class TestSentBucketCache(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out1, self.out2 = _get_example_data()

  def test_sentbucket_cache(self):
    cached_stats1 = compare_mt_main.generate_sentence_bucketed_report(self.ref, [self.out1], to_cache=True)
    cached_stats2 = compare_mt_main.generate_sentence_bucketed_report(self.ref, [self.out2], to_cache=True)
    self.assertTrue('stats' in cached_stats1)
    self.assertTrue('stats' in cached_stats2)
    ori_report = compare_mt_main.generate_sentence_bucketed_report(self.ref, [self.out1, self.out2])
    cached_report = compare_mt_main.generate_sentence_bucketed_report(self.ref, [self.out1, self.out2], cache_dicts=[cached_stats1, cached_stats2])
    self.assertTrue(cached_report.sys_stats == ori_report.sys_stats)

class TestNgramCache(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out1, self.out2 = _get_example_data()

  def test_ngram_cache(self):
    reporters.sys_names = [f'sys{i+1}' for i in range(2)]
    cached_stats1 = compare_mt_main.generate_ngram_report(self.ref, [self.out1], to_cache=True)
    cached_stats2 = compare_mt_main.generate_ngram_report(self.ref, [self.out2], to_cache=True)
    self.assertTrue('totals' in cached_stats1 and 'matches' in cached_stats1 and 'overs' in cached_stats1 and 'unders' in cached_stats1)
    self.assertTrue('totals' in cached_stats2 and 'matches' in cached_stats2 and 'overs' in cached_stats2 and 'unders' in cached_stats2)
    ori_report = compare_mt_main.generate_ngram_report(self.ref, [self.out1, self.out2])
    cached_report = compare_mt_main.generate_ngram_report(self.ref, [self.out1, self.out2], cache_dicts=[cached_stats1, cached_stats2])
    self.assertTrue(cached_report.scorelist == ori_report.scorelist)

class TestSentExamCache(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.ref, self.out1, self.out2 = _get_example_data()

  def test_sentexam_cache(self):
    reporters.sys_names = [f'sys{i+1}' for i in range(2)]
    cached_stats1 = compare_mt_main.generate_sentence_examples(self.ref, [self.out1], to_cache=True)
    cached_stats2 = compare_mt_main.generate_sentence_examples(self.ref, [self.out2], to_cache=True)
    self.assertTrue('scores' in cached_stats1 and 'strs' in cached_stats1)
    self.assertTrue('scores' in cached_stats2 and 'strs' in cached_stats2)
    ori_report = compare_mt_main.generate_sentence_examples(self.ref, [self.out1, self.out2])
    cached_report = compare_mt_main.generate_sentence_examples(self.ref, [self.out1, self.out2], cache_dicts=[cached_stats1, cached_stats2])
    self.assertTrue(cached_report.scorediff_lists== ori_report.scorediff_lists)

if __name__ == "__main__":
  unittest.main()
