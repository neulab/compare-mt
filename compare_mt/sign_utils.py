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


def eval_with_paired_bootstrap(ref, outs, src,
                               scorer,
                               compare_directions=[(0, 1)],
                               num_samples=1000, sample_ratio=0.5,
                               cache_stats=None):
  """
  Evaluate with paired boostrap.
  This compares several systems, performing a signifiance tests with
  paired bootstrap resampling to compare the accuracy of the specified systems.

  Args:
    ref: The correct labels
    outs: The output of systems
    src: The source corpus
    scorer: The scorer
    compare_directions: A string specifying which two systems to compare
    num_samples: The number of bootstrap samples to take
    sample_ratio: The ratio of samples to take every time
    cache_stats: The precomputed statistics

  Returns:
    A tuple containing the win ratios, statistics for systems
  """
  sys_scores = [[] for _ in outs] 
  wins = [[0, 0, 0] for _ in compare_directions] if compare_directions is not None else None
  n = len(ref)
  ids = list(range(n))

  if cache_stats is None:
    cache_stats = [scorer.cache_stats(ref, out, src=src) for out in outs]
  sample_size = int(n*sample_ratio)
  for _ in range(num_samples):
    # Subsample the gold and system outputs (with replacement)
    reduced_ids = np.random.choice(ids, size=sample_size, replace=True)
    # Calculate accuracy on the reduced sample and save stats
    if cache_stats[0]:
      sys_score, _ = zip(*[scorer.score_cached_corpus(reduced_ids, cache_stat) for cache_stat in cache_stats])
    else:
      reduced_ref = [ref[i] for i in reduced_ids]
      reduced_outs = [[out[i] for i in reduced_ids] for out in outs]
      reduced_src = [src[i] for i in reduced_ids]
      sys_score, _ = zip(*[scorer.score_corpus(reduced_ref, reduced_out, reduced_src) for reduced_out in reduced_outs])

    if wins is not None:
      for i, compare_direction in enumerate(compare_directions): 
        left, right = compare_direction
        if sys_score[left] > sys_score[right]:
          wins[i][0] += 1
        if sys_score[left] < sys_score[right]:
          wins[i][1] += 1
        else:
          wins[i][2] += 1
    
    for i in range(len(outs)): 
      sys_scores[i].append(sys_score[i])

  # Print win stats
  wins = [[x/float(num_samples) for x in win] for win in wins] if wins is not None else None

  # Print system stats
  sys_stats = []
  for i in range(len(outs)): 
    sys_scores[i].sort()
    sys_stats.append({
      'mean':np.mean(sys_scores[i]),
      'median':np.median(sys_scores[i]),
      'lower_bound':sys_scores[i][int(num_samples * 0.025)],
      'upper_bound':sys_scores[i][int(num_samples * 0.975)]
    })
 
  return wins, sys_stats
