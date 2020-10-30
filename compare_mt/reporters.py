import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt 
plt.rcParams['font.family'] = 'sans-serif'
import numpy as np
import os
import itertools
from compare_mt.formatting import fmt

from functools import partial
from http.server import SimpleHTTPRequestHandler, HTTPServer
import socket
from pathlib import Path
import logging as log

log.basicConfig(level=log.INFO)

# Global variables used by all reporters. These are set by compare_mt_main.py
sys_names = None
fig_size = None

# The CSS style file to use
css_style = """
html {
  font-family: sans-serif;
}

table, th, td {
  border: 1px solid black;
}

th, td {
  padding: 2px;
}

tr:hover {background-color: #f5f5f5;}

tr:nth-child(even) {background-color: #f2f2f2;}

th {
  background-color: #396AB1;
  color: white;
}

em {
  font-weight: bold;
}

caption {
  font-size: 14pt;
  font-weight: bold;
}

table {
  border-collapse: collapse;
}
"""

# The Javascript header to use
javascript_style = """
function showhide(elem) {
  var x = document.getElementById(elem);
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}
"""

fig_counter, tab_counter = 0, 0
def next_fig_id():
  global fig_counter
  fig_counter += 1
  return f'{fig_counter:03d}'
def next_tab_id():
  global tab_counter
  tab_counter += 1
  return f'{tab_counter:03d}'

bar_colors = ["#7293CB", "#E1974C", "#84BA5B", "#D35E60", "#808585", "#9067A7", "#AB6857", "#CCC210"]

def make_bar_chart(datas,
                   output_directory, output_fig_file, output_fig_format='png',
                   errs=None, title=None, xlabel=None, xticklabels=None, ylabel=None):
  fig, ax = plt.subplots(figsize=fig_size)
  ind = np.arange(len(datas[0]))
  width = 0.7/len(datas)
  bars = []
  for i, data in enumerate(datas):
    err = errs[i] if errs != None else None
    bars.append(ax.bar(ind+i*width, data, width, color=bar_colors[i], bottom=0, yerr=err))
  # Set axis/title labels
  if title is not None:
    ax.set_title(title)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  if xticklabels is not None:
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=70)
  else:
    ax.xaxis.set_visible(False) 

  ax.legend(bars, sys_names)
  ax.autoscale_view()

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
  out_file = os.path.join(output_directory, f'{output_fig_file}.{output_fig_format}')
  plt.savefig(out_file, format=output_fig_format, bbox_inches='tight')

def html_img_reference(fig_file, title):
  latex_code_pieces = [r"\begin{figure}[h]",
                       r"  \centering",
                       r"  \includegraphics{" + fig_file + ".pdf}",
                       r"  \caption{" + title + "}",
                       r"  \label{fig:" + fig_file + "}",
                       r"\end{figure}"]
  latex_code = "\n".join(latex_code_pieces)
  return (f'<img src="{fig_file}.png" alt="{title}"> <br/>' +
          f'<button onclick="showhide(\'{fig_file}_latex\')">Show/Hide LaTeX</button> <br/>' +
          f'<pre id="{fig_file}_latex" style="display:none">{latex_code}</pre>')

class Report: 
  # def __init__(self, iterable=(), **kwargs):
  #   # Initialize a report by a dictionary which contains all the statistics
  #   self.__dict__.update(iterable, **kwargs)
  
  def print(self): 
    raise NotImplementedError('print must be implemented in subclasses of Report')

  def plot(self, output_directory, output_fig_file, output_fig_type):
    raise NotImplementedError('plot must be implemented in subclasses of Report')

  def print_header(self, header):
    print(f'********************** {header} ************************')

  def print_tabbed_table(self, tab):
    for x in tab:
      print('\t'.join([fmt(y, latex=False) if y else '' for y in x]))
    print()

  def generate_report(self, output_fig_file=None, output_fig_format=None, output_directory=None):
    self.print()

class ScoreReport(Report):
  def __init__(self, scorer, scores, strs,
               wins=None, sys_stats=None, prob_thresh=0.05,
               title=None):
    self.scorer = scorer 
    self.scores = scores
    self.strs = [f'{fmt(x)} ({y})' if y else fmt(x) for (x,y) in zip(scores,strs)]
    self.wins = wins
    self.sys_stats = sys_stats
    self.output_fig_file = f'{next_fig_id()}-score-{scorer.idstr()}'
    self.prob_thresh = prob_thresh
    self.title = scorer.name() if not title else title

  def winstr_pval(self, my_wins):
    if 1-my_wins[0] < self.prob_thresh:
      winstr = 's1>s2'
    elif 1-my_wins[1] < self.prob_thresh:
      winstr = 's2>s1'
    else:
      winstr = '-'
    pval = 1-(my_wins[0] if my_wins[0] > my_wins[1] else my_wins[1])
    return winstr, pval

  def scores_to_tables(self):
    if self.wins is None:
      # Single table with just scores
      return [[""]+sys_names, [self.scorer.name()]+self.strs], None
    elif len(self.scores) == 1:
      # Single table with scores for one system
      return [
        [""]+sys_names,
        [self.scorer.name()]+self.strs,
        [""]+[f'[{fmt(x["lower_bound"])},{fmt(x["upper_bound"])}]' for x in self.sys_stats]
      ], None
    elif len(self.scores) == 2:
      # Single table with scores and wins for two systems
      winstr, pval = self.winstr_pval(self.wins[0][1])
      return [
        [""]+sys_names+["Win?"],
        [self.scorer.name()]+self.strs+[winstr],
        [""]+[f'[{fmt(x["lower_bound"])},{fmt(x["upper_bound"])}]' for x in self.sys_stats]+[f'p={fmt(pval)}']
      ], None
    else:
      # Table with scores, and separate one with wins for multiple systems
      wptable = [['v s1 / s2 ->'] + [sys_names[i] for i in range(1,len(self.scores))]]
      for i in range(0, len(self.scores)-1):
        wptable.append([sys_names[i]] + [""] * (len(self.scores)-1))
      for (left,right), my_wins in self.wins:
        winstr, pval = self.winstr_pval(my_wins)
        wptable[left+1][right] = f'{winstr} (p={fmt(pval)})'
      return [[""]+sys_names, [self.scorer.name()]+self.strs], wptable

  def print(self):
    aggregate_table, win_table = self.scores_to_tables()
    self.print_header('Aggregate Scores')
    print(f'{self.title}:')
    self.print_tabbed_table(aggregate_table)
    if win_table:
      self.print_tabbed_table(win_table)

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    sys = [[score] for score in self.scores]
    if self.wins:
      sys_errs = [np.array([ [score-stat['lower_bound']], [stat['upper_bound']-score] ]) for (score,stat) in zip(self.scores, self.sys_stats)]
    else:
      sys_errs = None
    xticklabels = None

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   output_fig_format=output_fig_format,
                   errs=sys_errs, ylabel=self.scorer.name(),
                   xticklabels=xticklabels)

  def html_content(self, output_directory):
    aggregate_table, win_table = self.scores_to_tables()
    html = html_table(aggregate_table, title=self.title)
    if win_table:
      html += html_table(win_table, title=f'{self.scorer.name()} Wins')
    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Score Comparison')
    return html
    
class WordReport(Report):
  def __init__(self, bucketer, statistics,
               acc_type, header,
               examples=None,
               bucket_cnts=None,
               bucket_intervals=None,
               src_sents=None,
               ref_sents=None, ref_labels=None,
               out_sents=None, out_labels=None,
               src_labels=None, ref_aligns=None,
               title=None):
    self.bucketer = bucketer
    self.statistics = [[s for s in stat] for stat in statistics]
    self.examples = examples
    self.bucket_cnts = bucket_cnts
    self.bucket_intervals = bucket_intervals
    self.src_sents = src_sents
    self.ref_sents = ref_sents
    self.ref_labels = ref_labels
    self.out_sents = out_sents
    self.out_labels = out_labels
    self.src_labels = src_labels
    self.ref_aligns = ref_aligns
    self.acc_type = acc_type
    self.header = header
    self.acc_type_map = {'prec': 3, 'rec': 4, 'fmeas': 5}
    self.output_fig_file = f'{next_fig_id()}-wordacc-{bucketer.name()}'
    self.title = title if title else f'word {acc_type} by {bucketer.name()} bucket'

  def print(self):
    acc_type_map = self.acc_type_map
    bucketer, statistics, acc_type, header = self.bucketer, self.statistics, self.acc_type, self.header
    self.print_header(header)
    acc_types = acc_type.split('+')
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      print(f'--- {self.title}')
      # first line
      print(f'{bucketer.name()}', end='')
      if self.bucket_cnts is not None:
        print(f'\t# words', end='')
      for sn in sys_names:
        print(f'\t{sn}', end='')
      print()
      # stats
      for i, bucket_str in enumerate(bucketer.bucket_strs):
        print(f'{bucket_str}', end='')
        if self.bucket_cnts is not None:
          print(f'\t{self.bucket_cnts[i]}', end='')
        for j, match in enumerate(statistics):
          print(f'\t{fmt(match[i][aid])}', end='')
          if self.bucket_intervals is not None:
            low, up = self.bucket_intervals[j][i][aid]
            print(f' [{fmt(low)}, {fmt(up)}]', end='')
        print()
      print()

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    acc_types = self.acc_type.split('+')
    for at in acc_types:
      if at not in self.acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = self.acc_type_map[at]
      sys = [[m[aid] for m in match] for match in self.statistics]
      xticklabels = [s for s in self.bucketer.bucket_strs] 

      if self.bucket_intervals:
        errs = []
        for i, match in enumerate(sys):
          lows, ups = [], []
          for j, score in enumerate(match):
            low, up = self.bucket_intervals[i][j][aid] 
            lows.append(score-low)
            ups.append(up-score)
          errs.append(np.array([lows, ups]) )
      else:
        errs = None

      make_bar_chart(sys,
                     output_directory, output_fig_file,
                     output_fig_format=output_fig_format,
                     errs=errs,
                     xlabel=self.bucketer.name(), ylabel=at,
                     xticklabels=xticklabels)

  def highlight_words(self, sent, hls=None):
    if not hls:
      return ' '.join(sent)
    return ' '.join([f'<em>{w}</em>' if hl else w for (w,hl) in zip(sent, hls)])

  def write_examples(self, title, output_directory):
    # Create separate examples HTML file
    html = ''
    for bi, bucket_examples in enumerate(self.examples):
      html += f'<a name="bucket{bi}"/>'
      html += tag_str('h3', f'Examples for Bucket {self.bucketer.bucket_strs[bi]}')
      for tag, examp_ids in bucket_examples:
        #  Skip ones with no examples
        if len(examp_ids) == 0:
          continue
        html += tag_str('h4', tag)
        for eid in examp_ids:
          table = [['', 'Output']]
          # Find buckets for the examples if it's on the source side (will have alignments in this case)
          if self.ref_aligns:
            _, _, _, src_buckets, ref_aligns, ref_matches = \
              self.bucketer._calc_src_buckets_and_matches(self.src_sents[eid],
                                                          self.src_labels[eid] if self.src_labels else None,
                                                          self.ref_sents[eid],
                                                          self.ref_aligns[eid],
                                                          [x[eid] for x in self.out_sents])
            src_hls = [x == bi for x in src_buckets]
            table.append(['Src', self.highlight_words(self.src_sents[eid], src_hls)])
            ref_hls = [False for _ in self.ref_sents[eid]]
            out_hls = [[False for _ in x[eid]] for x in self.out_sents]
            for sid, tid in self.ref_aligns[eid]:
              if src_hls[sid]:
                ref_hls[tid] = True
                for rm, ohls in zip(ref_matches, out_hls):
                  if rm[tid] >= 0:
                    ohls[rm[tid]] = True
          # Find buckets for the examples if it's on the target side
          else:
            _, _, _, ref_buckets, out_buckets, out_matches = \
              self.bucketer._calc_trg_buckets_and_matches(self.ref_sents[eid],
                                                          self.ref_labels[eid] if self.ref_labels else None,
                                                          [x[eid] for x in self.out_sents],
                                                          [x[eid] for x in self.out_labels] if self.out_labels else None)
            ref_hls = [x == bi for x in ref_buckets]
            out_hls = [[(b == bi and m >= 0) for (b,m) in zip(ob, om)] for (ob, om) in zip(out_buckets, out_matches)]
          table.append(['Ref', self.highlight_words(self.ref_sents[eid], ref_hls)])
          for sn, oss, ohl in itertools.zip_longest(sys_names, self.out_sents, out_hls):
            table.append([sn, self.highlight_words(oss[eid], ohl)])
          html += html_table(table, None)
    with open(f'{output_directory}/{self.output_fig_file}.html', 'w') as example_stream:
      example_stream.write(styled_html_message(title, html))

  def html_content(self, output_directory):
    acc_type_map = self.acc_type_map
    bucketer, matches, acc_type, header = self.bucketer, self.statistics, self.acc_type, self.header
    acc_types = acc_type.split('+')

    title = f'Word {acc_type} by {bucketer.name()} bucket' if not self.title else self.title

    if self.examples:
      self.write_examples(title, output_directory)

    # Create main HTML content
    html = ''
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      line = [bucketer.name()]
      if self.bucket_cnts is not None:
        line.append('# words')
      line += sys_names
      table = [line]
      if self.examples:
        table[0].append('Examples')
      for i, bs in enumerate(bucketer.bucket_strs):
        line = [bs]
        if self.bucket_cnts is not None:
          line.append(f'{self.bucket_cnts[i]}')
        for j, match in enumerate(matches):
          line.append(f'{fmt(match[i][aid])}')
          if self.bucket_intervals is not None:
            low, up = self.bucket_intervals[j][i][aid]
            line[-1] += f'<font size=2> [{fmt(low)}, {fmt(up)}]</font>'
        if self.examples:
          line.append(f'<a href="{self.output_fig_file}.html#bucket{i}">Examples</a>')
        table += [line] 
      html += html_table(table, title, latex_ignore_cols={3})
      img_name = f'{self.output_fig_file}-{at}'
      for ext in ('png', 'pdf'):
        self.plot(output_directory, img_name, ext)
      html += html_img_reference(img_name, self.header)
    return html 

class NgramReport(Report):
  def __init__(self, scorelist, report_length, min_ngram_length, max_ngram_length,
               matches, compare_type, alpha, compare_directions=[(0, 1)], label_files=None, title=None):
    self.scorelist = scorelist
    self.report_length = report_length 
    self.min_ngram_length = min_ngram_length
    self.max_ngram_length = max_ngram_length
    self.matches = matches
    self.compare_type = compare_type
    self.label_files = label_files
    self.alpha = alpha
    self.compare_directions = compare_directions
    self.title = title

  def print(self):
    report_length = self.report_length
    self.print_header('N-gram Difference Analysis')
    if self.title:
      print(f'--- {self.title}')
    else:
      print(f'--- min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
      print(f'    report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')

    if self.label_files is not None:
      print(self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      print(f'--- {report_length} n-grams where {sys_names[left]}>{sys_names[right]} in {self.compare_type}')
      for k, v in self.scorelist[i][:report_length]:
        print(f"{' '.join(k)}\t{fmt(v)} (sys{left+1}={self.matches[left][k]}, sys{right+1}={self.matches[right][k]})")
      print()
      print(f'--- {report_length} n-grams where {sys_names[right]}>{sys_names[left]} in {self.compare_type}')
      for k, v in reversed(self.scorelist[i][-report_length:]):
        print(f"{' '.join(k)}\t{fmt(v)} (sys{left+1}={self.matches[left][k]}, sys{right+1}={self.matches[right][k]})")
      print()

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    raise NotImplementedError('Plotting is not implemented for n-gram reports')

  def html_content(self, output_directory=None):
    report_length = self.report_length
    if self.title:
      html = tag_str('p', self.title)
    else:
      html = tag_str('p', f'min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
      html += tag_str('p', f'report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')
      if self.label_files is not None:
        html += tag_str('p', self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      title = f'{report_length} n-grams where {sys_names[left]}>{sys_names[right]} in {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{sys_names[left]}', f'{sys_names[right]}']]
      table.extend([[' '.join(k), fmt(v), self.matches[left][k], self.matches[right][k]] for k, v in self.scorelist[i][:report_length]])
      html += html_table(table, title)

      title = f'{report_length} n-grams where {sys_names[right]}>{sys_names[left]} in {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{sys_names[left]}', f'{sys_names[right]}']]
      table.extend([[' '.join(k), fmt(v), self.matches[left][k], self.matches[right][k]] for k, v in reversed(self.scorelist[i][-report_length:])])
      html += html_table(table, title)
    return html 

class SentenceReport(Report):

  def __init__(self, bucketer=None, sys_stats=None, statistic_type=None, scorer=None, bucket_cnts=None, bucket_intervals=None, title=None):
    self.bucketer = bucketer
    self.sys_stats = [[s for s in stat] for stat in sys_stats]
    self.statistic_type = statistic_type
    self.scorer = scorer
    self.bucket_cnts = bucket_cnts
    self.bucket_intervals = bucket_intervals
    self.yname = scorer.name() if statistic_type == 'score' else statistic_type
    self.yidstr = scorer.idstr() if statistic_type == 'score' else statistic_type
    self.output_fig_file = f'{next_fig_id()}-sent-{bucketer.idstr()}-{self.yidstr}'
    if title:
      self.title = title
    elif scorer:
      self.title = f'bucket type: {bucketer.name()}, statistic type: {scorer.name()}'
    else:
      self.title = f'bucket type: {bucketer.name()}, statistic type: {statistic_type}'

  def print(self):
    self.print_header('Sentence Bucket Analysis')
    print(f'--- {self.title}')
    # first line
    print(f'{self.bucketer.idstr()}', end='')
    if self.bucket_cnts is not None:
      print(f'\t# sents', end='')
    for sn in sys_names:
      print(f'\t{sn}', end='')
    print()
    for i, bs in enumerate(self.bucketer.bucket_strs):
      print(f'{bs}', end='')
      if self.bucket_cnts is not None:
        print(f'\t{self.bucket_cnts[i]}', end='')
      for j, stat in enumerate(self.sys_stats):
        print(f'\t{fmt(stat[i])}', end='')
        if self.bucket_intervals is not None:
          interval =  self.bucket_intervals[j][i]
          low, up = interval['lower_bound'], interval['upper_bound']
          print(f' [{fmt(low)}, {fmt(up)}]', end='')
      print()
    print()

  def plot(self, output_directory='outputs', output_fig_file='word-acc', output_fig_format='pdf'):
    sys = self.sys_stats
    xticklabels = [s for s in self.bucketer.bucket_strs] 

    if self.bucket_intervals:
      errs = []
      for i, stat in enumerate(sys):
        lows, ups = [], []
        for j, score in enumerate(stat):
          interval = self.bucket_intervals[i][j]
          low, up = interval['lower_bound'], interval['upper_bound']
          lows.append(score-low)
          ups.append(up-score)
        errs.append(np.array([lows, ups]) )
    else:
      errs = None

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   output_fig_format=output_fig_format,
                   errs=errs,
                   xlabel=self.bucketer.name(), ylabel=self.yname,
                   xticklabels=xticklabels)

  def html_content(self, output_directory=None):
    line = [self.bucketer.idstr()]
    if self.bucket_cnts is not None:
      line.append('# sents')
    line += sys_names
    table = [ line ]
    for i, bs in enumerate(self.bucketer.bucket_strs):
      line = [bs]
      if self.bucket_cnts is not None:
        line.append(f'\t{self.bucket_cnts[i]}')
      for j, stat in enumerate(self.sys_stats):
        line.append(fmt(stat[i]))
        if self.bucket_intervals is not None:
          interval =  self.bucket_intervals[j][i]
          low, up = interval['lower_bound'], interval['upper_bound']
          line[-1] += f'<font size=2> [{fmt(low)}, {fmt(up)}]</font>'
      table.extend([line])
    html = html_table(table, self.title)
    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Sentence Bucket Analysis')
    return html 

class SentenceExampleReport(Report):

  def __init__(self, report_length=None, scorediff_lists=None, scorer=None, ref=None, outs=None, src=None, compare_directions=[(0, 1)], title=None):
    self.report_length = report_length 
    self.scorediff_lists = scorediff_lists
    self.scorer = scorer
    self.ref = ref
    self.outs = outs
    self.src = src
    self.compare_directions = compare_directions
    self.title = title

  def print(self):
    self.print_header('Sentence Examples Analysis')
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      sleft, sright = sys_names[left], sys_names[right]
      print(f'--- {report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        print(f"{sleft}-{sright}={fmt(-bdiff)}, {sleft}={fmt(s1)}, {sright}={fmt(s2)}")
        if self.src and self.src[i]:
          print(f"Src:  {' '.join(self.src[i])}")
        print ( 
          f"Ref:  {' '.join(ref[i])}\n"
          f"{sleft}: {' '.join(out1[i])}\n"
          f"{sright}: {' '.join(out2[i])}\n"
        )

      print(f'--- {report_length} sentences where {sright}>{sleft} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        print(f"{sleft}-{sright}={fmt(-bdiff)}, {sleft}={fmt(s1)}, {sright}={fmt(s2)}")
        if self.src and self.src[i]:
          print(f"Src:  {' '.join(self.src[i])}")
        print (
          f"Ref:  {' '.join(ref[i])}\n"
          f"{sleft}: {' '.join(out1[i])}\n"
          f"{sright}: {' '.join(out2[i])}\n"
        )

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    pass 

  def html_content(self, output_directory=None):
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      sleft, sright = sys_names[left], sys_names[right]
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      html = tag_str('h4', f'{report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        table = [['', 'Output', f'{self.scorer.idstr()}']]
        if self.src and self.src[i]:
          table.append(['Src', ' '.join(self.src[i]), ''])
        table += [
          ['Ref', ' '.join(ref[i]), ''],
          [f'{sleft}', ' '.join(out1[i]), fmt(s1)],
          [f'{sright}', ' '.join(out2[i]), fmt(s2)]
        ]
        
        html += html_table(table, None)

      html += tag_str('h4', f'{report_length} sentences where {sright}>{sleft} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        table = [['', 'Output', f'{self.scorer.idstr()}']]
        if self.src and self.src[i]:
          table.append(['Src', ' '.join(self.src[i]), ''])
        table += [
          ['Ref', ' '.join(ref[i]), ''],
          [f'{sleft}', ' '.join(out1[i]), fmt(s1)],
          [f'{sright}', ' '.join(out2[i]), fmt(s2)]
        ]

        html += html_table(table, None)

    return html


def tag_str(tag, str, new_line=''):
  return f'<{tag}>{new_line} {str} {new_line}</{tag}>'

def html_table(table, title=None, bold_rows=1, bold_cols=1, latex_ignore_cols={}):
  html = '<table border="1">\n'
  if title is not None:
    html += tag_str('caption', title)
  for i, row in enumerate(table):
    tag_type = 'th' if (i < bold_rows) else 'td'
    table_row = '\n  '.join(tag_str('th' if j < bold_cols else tag_type, rdata) for (j, rdata) in enumerate(row))
    html += tag_str('tr', table_row)
  html += '\n</table>\n <br/>'

  tab_id = next_tab_id()
  latex_code = "\\begin{table}[t]\n  \\centering\n"
  cs = ['c'] * len(table[0])
  if bold_cols != 0:
    cs[bold_cols-1] = 'c||'
  latex_code += "  \\begin{tabular}{"+''.join(cs)+"}\n"
  for i, row in enumerate(table):
    latex_code += ' & '.join([fmt(x) for c_i, x in enumerate(row) if c_i not in latex_ignore_cols]) + (' \\\\\n' if i != bold_rows-1 else ' \\\\ \\hline \\hline\n')
  latex_code += "  \\end{tabular}\n  \\caption{Caption}\n  \\label{tab:table"+tab_id+"}\n\\end{table}"

  html += (f'<button onclick="showhide(\'{tab_id}_latex\')">Show/Hide LaTeX</button> <br/>' +
           f'<pre id="{tab_id}_latex" style="display:none">{latex_code}</pre>')
  return html

def styled_html_message(report_title, content):
  content = content.encode("ascii","xmlcharrefreplace").decode()
  return (f'<html>\n<head>\n<link rel="stylesheet" href="compare_mt.css">\n</head>\n'+
          f'<script>\n{javascript_style}\n</script>\n'+
          f'<body>\n<h1>{report_title}</h1>\n {content} \n</body>\n</html>')

def generate_html_report(reports, output_directory, report_title):
  content = []
  for name, rep in reports:
    content.append(f'<h2>{name}</h2>')
    for r in rep:
      content.append(r.html_content(output_directory))
  content = "\n".join(content)
  
  if not os.path.exists(output_directory):
        os.makedirs(output_directory)
  html_file = os.path.join(output_directory, 'index.html')
  with open(html_file, 'w') as f:
    f.write(styled_html_message(report_title, content))
  css_file = os.path.join(output_directory, 'compare_mt.css')
  with open(css_file, 'w') as f:
    f.write(css_style)

def launch_http_server(output_directory: str, bind_address:str ='0.0.0.0', bind_port: int=8000):
  assert Path(output_directory).is_dir()
  hostname = bind_address if bind_address != '0.0.0.0' else socket.gethostname()
  log.info(f'Directory = {output_directory}')
  log.info(f'Launching a web server:: http://{hostname}:{bind_port}/')
  Handler = partial(SimpleHTTPRequestHandler, directory=output_directory)
  server = HTTPServer(server_address=(bind_address, bind_port),
                               RequestHandlerClass=Handler)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass # all good! Exiting without printing stacktrace
  
