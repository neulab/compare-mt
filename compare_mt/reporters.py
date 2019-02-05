import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt 
plt.rcParams['font.family'] = 'sans-serif'
import numpy as np
import os

# Global variables used by all reporters. These are set by compare_mt_main.py
sys_names = None

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
  fig, ax = plt.subplots() 
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
  latex_code = (
    "\\begin{figure}[h]\n"+
    "  \centering\n"+
    "  \includegraphics{"+fig_file+".pdf}\n"+
    "  \caption{"+title+"}\n"+
    "  \label{fig:"+fig_file+"}\n"+
    "\end{figure}"
  )
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
      print('\t'.join([str(y) if y else '' for y in x]))
    print()

  def generate_report(self, output_fig_file=None, output_fig_format=None, output_directory=None):
    self.print()

class ScoreReport(Report):
  def __init__(self, scorer, scores, strs,
               wins=None, sys_stats=None):
    self.scorer = scorer 
    self.scores = scores
    self.strs = [f'{x:.4f} ({y})' if y else f'{x:.4f}' for (x,y) in zip(scores,strs)]
    self.wins = wins
    self.sys_stats = sys_stats
    self.output_fig_file = f'{next_fig_id()}-score-{scorer.idstr()}'
    self.prob_thresh = 0.05

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
    elif len(self.scores) == 2:
      # Single table with scores and wins for two systems
      winstr, pval = self.winstr_pval(self.wins[0][1])
      return [
        [""]+sys_names+["Win?"],
        [self.scorer.name()]+self.strs+[winstr],
        [""]+[f'[{x["lower_bound"]:.4f},{x["upper_bound"]:.4f}]' for x in self.sys_stats]+[f'p={pval:.4f}']
      ], None
    else:
      # Table with scores, and separate one with wins for multiple systems
      wptable = [['v s1 / s2 ->'] + [sys_names[i] for i in range(1,len(self.scores))]]
      for i in range(0, len(self.scores)-1):
        wptable.append([sys_names[i]] + [""] * (len(self.scores)-1))
      for (left,right), my_wins in self.wins:
        winstr, pval = self.winstr_pval(my_wins)
        wptable[left+1][right] = f'{winstr} (p={pval:.4f})'
      return [[""]+sys_names, [self.scorer.name()]+self.strs], wptable

  def print(self):
    aggregate_table, win_table = self.scores_to_tables()
    self.print_header('Aggregate Scores')
    print(f'{self.scorer.name()}:')
    self.print_tabbed_table(aggregate_table)
    if win_table:
      self.print_tabbed_table(win_table)

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    sys = [[score] for score in self.scores]
    if self.wins:
      sys_errs = [np.array([[score-stat['lower_bound'], stat['upper_bound']-score]]) for (score,stat) in zip(self.scores, self.sys_stats)]
    xticklabels = None

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   output_fig_format=output_fig_format,
                   errs=sys_errs, title='Score Comparison', ylabel=self.scorer.name(),
                   xticklabels=xticklabels)

  def html_content(self, output_directory):
    aggregate_table, win_table = self.scores_to_tables()
    html = html_table(aggregate_table, caption=self.scorer.name())
    if win_table:
      html += html_table(win_table, caption=f'{self.scorer.name()} Wins')
    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Score Comparison')
    return html
    
class WordReport(Report):
  def __init__(self, bucketer, matches, acc_type, header):
    self.bucketer = bucketer
    self.matches = [[m for m in match] for match in matches]
    self.acc_type = acc_type
    self.header = header
    self.acc_type_map = {'prec': 3, 'rec': 4, 'fmeas': 5}
    self.output_fig_file = f'{next_fig_id()}-wordacc-{bucketer.name()}'

  def print(self):
    acc_type_map = self.acc_type_map
    bucketer, matches, acc_type, header = self.bucketer, self.matches, self.acc_type, self.header
    self.print_header(header)
    acc_types = acc_type.split('+')
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      print(f'--- word {acc_type} by {bucketer.name()} bucket')
      for i, bucket_str in enumerate(bucketer.bucket_strs):
        print(f'{bucket_str}', end='')
        for match in matches:
          print(f'\t{match[i][aid]:.4f}', end='')
        print()
      print()

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    acc_types = self.acc_type.split('+')
    for at in acc_types:
      if at not in self.acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = self.acc_type_map[at]
      sys = [[m[aid] for m in match] for match in self.matches]
      xticklabels = [s for s in self.bucketer.bucket_strs] 

      make_bar_chart(sys,
                     output_directory, output_fig_file,
                     output_fig_format=output_fig_format,
                     title='Word Accuracy Comparison', xlabel=self.bucketer.name(), ylabel=at,
                     xticklabels=xticklabels)
    
  def html_content(self, output_directory):
    acc_type_map = self.acc_type_map
    bucketer, matches, acc_type, header = self.bucketer, self.matches, self.acc_type, self.header
    acc_types = acc_type.split('+') 

    html = ''
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      caption = f'Word {acc_type} by {bucketer.name()} bucket'
      table = [[bucketer.name()] + sys_names]
      for i, bs in enumerate(bucketer.bucket_strs):
        line = [bs]
        for match in matches:
          line.append(f'{match[i][aid]:.4f}')
        table += [line] 
      html += html_table(table, caption)
      img_name = f'{self.output_fig_file}-{at}'
      for ext in ('png', 'pdf'):
        self.plot(output_directory, img_name, ext)
      html += html_img_reference(img_name, self.header)
    return html 

class NgramReport(Report):
  def __init__(self, scorelist, report_length, min_ngram_length, max_ngram_length,
               matches, compare_type, alpha, compare_directions=[(0, 1)], label_files=None):
    self.scorelist = scorelist
    self.report_length = report_length 
    self.min_ngram_length = min_ngram_length
    self.max_ngram_length = max_ngram_length
    self.matches = matches
    self.compare_type = compare_type
    self.label_files = label_files
    self.alpha = alpha
    self.compare_directions = compare_directions

  def print(self):
    report_length = self.report_length
    self.print_header('N-gram Difference Analysis')
    print(f'--- min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
    print(f'    report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')
    if self.label_files is not None:
      print(self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      print(f'--- {report_length} n-grams that {sys_names[left]} had higher {self.compare_type}')
      for k, v in self.scorelist[i][:report_length]:
        print('{}\t{} (sys{}={}, sys{}={})'.format(' '.join(k), v, left+1, self.matches[left][k], right+1, self.matches[right][k]))
      print(f'\n--- {report_length} n-grams that {sys_names[right]} had higher {self.compare_type}')
      for k, v in reversed(self.scorelist[i][-report_length:]):
        print('{}\t{} (sys{}={}, sys{}={})'.format(' '.join(k), v, left+1, self.matches[left][k], right+1, self.matches[right][k]))
      print()

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    raise NotImplementedError('Plotting is not implemented for n-gram reports')

  def html_content(self, output_directory=None):
    report_length = self.report_length
    html = tag_str('p', f'min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
    html += tag_str('p', f'report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')
    if self.label_files is not None:
      html += tag_str('p', self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      caption = f'{report_length} n-grams that {sys_names[left]} had higher {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{sys_names[left]}', f'{sys_names[right]}']]
      table.extend([[' '.join(k), f'{v:.2f}', self.matches[left][k], self.matches[right][k]] for k, v in self.scorelist[i][:report_length]])
      html += html_table(table, caption)

      caption = f'{report_length} n-grams that {sys_names[right]} had higher {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{sys_names[left]}', f'{sys_names[right]}']]
      table.extend([[' '.join(k), f'{v:.2f}', self.matches[left][k], self.matches[right][k]] for k, v in reversed(self.scorelist[i][-report_length:])])
      html += html_table(table, caption)
    return html 

class SentenceReport(Report):

  def __init__(self, bucketer=None, sys_stats=None, statistic_type=None, scorer=None):
    self.bucketer = bucketer
    self.sys_stats = [[s for s in stat] for stat in sys_stats]
    self.statistic_type = statistic_type
    self.scorer = scorer
    self.yname = scorer.name() if statistic_type == 'score' else statistic_type
    self.yidstr = scorer.idstr() if statistic_type == 'score' else statistic_type
    self.output_fig_file = f'{next_fig_id()}-sent-{bucketer.idstr()}-{self.yidstr}'
    if scorer:
      self.description = f'bucket type: {bucketer.name()}, statistic type: {scorer.name()}'
    else:
      self.description = f'bucket type: {bucketer.name()}, statistic type: {statistic_type}'

  def print(self):
    self.print_header('Sentence Bucket Analysis')
    print(f'--- {self.description}')
    for i, bs in enumerate(self.bucketer.bucket_strs):
      print(f'{bs}', end='')
      for stat in self.sys_stats:
        print(f'\t{stat[i]}', end='')
      print()
    print()

  def plot(self, output_directory='outputs', output_fig_file='word-acc', output_fig_format='pdf'):
    sys = self.sys_stats
    xticklabels = [s for s in self.bucketer.bucket_strs] 

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   output_fig_format=output_fig_format,
                   title='Sentence Bucket Analysis', xlabel=self.bucketer.name(), ylabel=self.yname,
                   xticklabels=xticklabels)

  def html_content(self, output_directory=None):
    table = [ [self.bucketer.name()] + sys_names ]
    for i, bs in enumerate(self.bucketer.bucket_strs):
      line = [bs]
      for stat in self.sys_stats:
        line.append(f'{stat[i]:.4f}')
      table.extend([line])
    html = html_table(table, self.description)
    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Sentence Bucket Analysis')
    return html 

class SentenceExampleReport(Report):

  def __init__(self, report_length=None, scorediff_lists=None, scorer=None, ref=None, outs=None, compare_directions=[(0, 1)]):
    self.report_length = report_length 
    self.scorediff_lists = scorediff_lists
    self.scorer = scorer
    self.ref = ref
    self.outs = outs
    self.compare_directions = compare_directions

  def print(self):
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      sleft, sright = sys_names[left], sys_names[right]
      print(f'--- {report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        print ('{}-{}={}, {}={}, {}={}\nRef:  {}\n{}: {}\n{}: {}\n'.format(sleft, sright, -bdiff, sleft, s1, sright, s2, ' '.join(ref[i]), sleft, ' '.join(out1[i]), sright, ' '.join(out2[i])))
      print(f'--- {report_length} sentences where {sright}>{sleft} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        print ('{}-{}={}, {}={}, {}={}\nRef:  {}\n{}: {}\n{}: {}\n'.format(sright, sleft, bdiff, sleft, s1, sright, s2, ' '.join(ref[i]), sleft, ' '.join(out1[i]), sright, ' '.join(out2[i])))

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    pass 

  def html_content(self, output_directory=None):
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      sleft, sright = sys_names[left], sys_names[right]
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      html = tag_str('h4', f'{report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        table = [
          ['', 'Output', f'{self.scorer.idstr()}'],
          ['Ref', ' '.join(ref[i]), ''],
          [f'{sleft}', ' '.join(out1[i]), f'{s1:.4f}'],
          [f'{sright}', ' '.join(out2[i]), f'{s2:.4f}']
        ]
        html += html_table(table, None)

      html += tag_str('h4', f'{report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        table = [
          ['', 'Output', f'{self.scorer.idstr()}'],
          ['Ref', ' '.join(ref[i]), ''],
          [f'{sleft}', ' '.join(out1[i]), f'{s1:.4f}'],
          [f'{sright}', ' '.join(out2[i]), f'{s2:.4f}']
        ]
        html += html_table(table, None)

    return html


def tag_str(tag, str, new_line=''):
  return f'<{tag}>{new_line} {str} {new_line}</{tag}>'

def html_table(table, caption=None, bold_rows=1, bold_cols=1):
  html = '<table border="1">\n'
  if caption is not None:
    html += tag_str('caption', caption)
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
    latex_code += ' & '.join([str(x) for x in row]) + (' \\\\\n' if i != bold_rows-1 else ' \\\\ \\hline \\hline\n')
  latex_code += "  \\end{tabular}\n  \\caption{Caption}\n  \\label{tab:table"+tab_id+"}\n\\end{table}"

  html += (f'<button onclick="showhide(\'{tab_id}_latex\')">Show/Hide LaTeX</button> <br/>' +
           f'<pre id="{tab_id}_latex" style="display:none">{latex_code}</pre>')
  return html

def generate_html_report(reports, output_directory):
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
    message = (f'<html>\n<head>\n<link rel="stylesheet" href="compare_mt.css">\n</head>\n'+
               f'<script>\n{javascript_style}\n</script>\n'+
               f'<body>\n<h1>compare_mt.py Analysis Report</h1>\n {content} \n</body>\n</html>')
    f.write(message)
  css_file = os.path.join(output_directory, 'compare_mt.css')
  with open(css_file, 'w') as f:
    f.write(css_style)
  
