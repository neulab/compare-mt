import matplotlib
from matplotlib import pyplot as plt 
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
import numpy as np
import os


class Report: 
  def __init__(self, iterable=(), **kwargs):
    # Initialize a report by a dictionary which contains all the statistics
    self.__dict__.update(iterable, **kwargs)
  
  def print(self): 
    raise NotImplementedError('print must be implemented in subclasses of Report')

  def plot(self):
    raise NotImplementedError('plot must be implemented in subclasses of Report')

  def print_header(self, header):
    print(f'********************** {header} ************************')


class ScoreReport(Report):
  def print(self):
    self.print_header('Aggregate Scores')
    print(f'{self.scorer_name}:')
    if self.str1 is not None:
      print(f' Sys1: {self.score1} ({self.str1})\n Sys2: {self.score2} ({self.str2})')
    else:
      print(f' Sys1: {self.score1}\n Sys2: {self.score2}')

    if 'wins' in self.__dict__:
      print('Significance test.')
      wins, sys1_stats, sys2_stats = self.wins, self.sys1_stats, self.sys2_stats
      print('Win ratio: Sys1=%.3f, Sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
      if wins[0] > wins[1]:
        print('(Sys1 is superior with p value p=%.3f)\n' % (1-wins[0]))
      elif wins[1] > wins[0]:
        print('(Sys2 is superior with p value p=%.3f)\n' % (1-wins[1]))

      print('Sys1: mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
              (sys1_stats['mean'], sys1_stats['median'], sys1_stats['lower_bound'], sys1_stats['upper_bound']))
      print('Sys2: mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
              (sys2_stats['mean'], sys2_stats['median'], sys2_stats['lower_bound'], sys2_stats['upper_bound']))
    print()
    
  def plot(self, output_path='outputs/', fig_name='score.pdf'):
    fig, ax = plt.subplots() 
    width = 0.35
    if 'wins' in self.__dict__:
      wins, sys1_stats, sys2_stats = self.wins, self.sys1_stats, self.sys2_stats
      mean, lrb, urb = sys1_stats['mean'], sys1_stats['lower_bound'], sys1_stats['upper_bound']
      sys1 = [self.score1, mean, sys1_stats['median']]
      N = len(sys1)
      sys1_err = np.zeros((2, N))
      sys1_err[0, 1] = mean - lrb 
      sys1_err[1, 1] = urb - mean 
      mean, lrb, urb = sys2_stats['mean'], sys2_stats['lower_bound'], sys2_stats['upper_bound']
      sys2 = [self.score2, sys2_stats['mean'], sys2_stats['median']]
      sys2_err = np.zeros((2, N))
      sys2_err[0, 1] = mean - lrb 
      sys2_err[1, 1] = urb - mean
      xlabel = [self.scorer_name, 'Bootstrap Mean', 'Bootstrap Median']

    ax.set_title('Aggregate Scores')
    ind = np.arange(len(sys1))
    p1 = ax.bar(ind, sys1, width, color='r', bottom=0, yerr=sys1_err)
    p2 = ax.bar(ind+width, sys2, width, color='y', bottom=0, yerr=sys2_err)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xlabel)

    ax.legend((p1[0], p2[0]), ('Sys1', 'Sys2'))
    ax.autoscale_view()

    if not os.path.exists(output_path):
      os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, fig_name))


class WordReport(Report):
  def print(self):
    acc_type_map = {'prec': 3, 'rec': 4, 'fmeas': 5}
    bucketer, matches1, matches2, acc_type, header = self.bucketer, self.matches1, self.matches2, self.acc_type, self.header
    self.print_header(header)
    acc_types = acc_type.split('+')
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      print(f'--- word {acc_type} by {bucketer.name()} bucket')
      for bucket_str, match1, match2 in zip(bucketer.bucket_strs, matches1, matches2):
        print("{}\t{:.4f}\t{:.4f}".format(bucket_str, match1[aid], match2[aid]))
      print()

  def plot(self):
    pass 

class NgramReport(Report):
  def print(self):
    pass 
  def plot(self):
    pass 

class SentenceReport(Report):
  def print(self):
    bucketer, stats1, stats2, bucket_type, statistic_type, score_measure = self.bucketer, self.stats1, self.stats2, self.bucket_type, self.statistic_type, self.score_measure
    self.print_header('Sentence Bucket Analysis')
    print(f'--- bucket_type={bucket_type}, statistic_type={statistic_type}, score_measure={score_measure}')
    for bs, s1, s2 in zip(bucketer.bucket_strs, stats1, stats2):
      print(f'{bs}\t{s1}\t{s2}')
    print()

  def plot(self):
    pass 

def create_reporter_from_profile(profile,
                                 stats={}):
  if profile == 'score':
    return ScoreReport(stats)
  elif profile == 'word':
    d = WordReport(stats)
    return WordReport(stats)
  elif profile == 'ngram':
    return NgramReport(stats)
  elif profile == 'sentence':
    return SentenceReport(stats)
  else:
    raise ValueError(f'Invalid profile for scorer {profile}')