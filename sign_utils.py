########################################################################################
# Compare two systems using bootstrap resampling                                       #
#  adapted from https://github.com/neubig/util-scripts/blob/master/paired-bootstrap.py #
#                                                                                      #
# See, e.g. the following paper for references                                         #
#                                                                                      #
# Statistical Significance Tests for Machine Translation Evaluation                    #
# Philipp Koehn                                                                        #
# http://www.aclweb.org/anthology/W04-3250                                             #
#                                                                                      #
########################################################################################

import numpy as np
import scorers
import nltk

def sample_and_compare(gold, sys1, sys2, sample_ratio, 
                       ids, wins, sys1_scores, sys2_scores, 
                       scorer):
  # Subsample the gold and system outputs
  np.random.shuffle(ids)
  reduced_ids = ids[:int(len(ids)*sample_ratio)]
  reduced_gold = [gold[i] for i in reduced_ids]
  reduced_sys1 = [sys1[i] for i in reduced_ids]
  reduced_sys2 = [sys2[i] for i in reduced_ids]
  # Calculate accuracy on the reduced sample and save stats
  sys1_score, _ = scorer.score_corpus(reduced_gold, reduced_sys1)
  sys2_score, _ = scorer.score_corpus(reduced_gold, reduced_sys2)
  if sys1_score > sys2_score:
    wins[0] += 1
  elif sys1_score < sys2_score:
    wins[1] += 1
  else:
    wins[2] += 1
  sys1_scores.append(sys1_score)
  sys2_scores.append(sys2_score)

def eval_with_paired_bootstrap(gold, sys1, sys2,
                               scorer,
                               num_samples=1000, sample_ratio=0.5):
  ''' Evaluate with paired boostrap
  This compares two systems, performing a signifiance tests with
  paired bootstrap resampling to compare the accuracy of the two systems.
  
  :param gold: The correct labels
  :param sys1: The output of system 1
  :param sys2: The output of system 2
  :param scorer: The scorer
  :param num_samples: The number of bootstrap samples to take
  :param sample_ratio: The ratio of samples to take every time
  '''
  assert(len(gold) == len(sys1))
  assert(len(gold) == len(sys2))
  
  sys1_scores = []
  sys2_scores = []
  wins = [0, 0, 0]
  n = len(gold)
  ids = list(range(n))

  try:
    from tqdm import tqdm
    for _ in tqdm(range(num_samples)):
      sample_and_compare(gold, sys1, sys2, sample_ratio, ids, wins, sys1_scores, sys2_scores, scorer)
  except ImportError:
    print('Install tqdm to see progress meter!')
    for _ in range(num_samples):
      sample_and_compare(gold, sys1, sys2, sample_ratio, ids, wins, sys1_scores, sys2_scores, scorer)

  # Print win stats
  wins = [x/float(num_samples) for x in wins]

  # Print system stats
  sys1_scores.sort()
  sys2_scores.sort()
 
  sys1_stats = {'mean':np.mean(sys1_scores), 'median':np.median(sys1_scores), 'lower_bound':sys1_scores[int(num_samples * 0.025)], 'upper_bound':sys1_scores[int(num_samples * 0.975)]}
  sys2_stats = {'mean':np.mean(sys2_scores), 'median':np.median(sys2_scores), 'lower_bound':sys2_scores[int(num_samples * 0.025)], 'upper_bound':sys2_scores[int(num_samples * 0.975)]}

  return wins, sys1_stats, sys2_stats

