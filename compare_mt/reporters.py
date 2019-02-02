import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt 
plt.rcParams['font.family'] = 'sans-serif'
import numpy as np
import os

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

bar_colors = ["#7293CB", "#E1974C", "#84BA5B", "#D35E60", "#808585", "#9067A7", "#AB6857", "#CCC210"]

def make_bar_chart(datas,
                   output_directory, output_fig_file, output_fig_format='png',
                   errs=None, sysnames=None, title=None, xlabel=None, xticklabels=None, ylabel=None):

  fig, ax = plt.subplots() 
  ind = np.arange(len(datas[0]))
  width = 0.7/len(datas)
  bars = []
  legend_handles = []
  legend_labels = []
  for i, data in enumerate(datas):
    err = errs[i] if errs != None else None
    bars.append(ax.bar(ind+i*width, data, width, color=bar_colors[i], bottom=0, yerr=err))
  ax.set_xticks(ind + width / 2)
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
  else:
    ax.xaxis.set_visible(False) 

  if sysnames is None:
    sysnames = [f'Sys{i}' for (i, x) in enumerate(datas)]
  ax.legend(bars, sysnames)
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

  def generate_report(self, output_fig_file=None, output_fig_format=None, output_directory=None):
    self.print()

    if output_fig_file is not None:
      self.plot(output_directory, output_fig_file, output_fig_format)

class ScoreReport(Report):
  def __init__(self, scorer_name, scores, strs, 
               wins=None, sys_stats=None, compare_directions=None):
    self.scorer_name = scorer_name 
    self.scores = scores
    self.strs = strs
    self.wins = wins
    self.sys_stats = sys_stats 
    self.compare_directions = compare_directions
    self.output_fig_file = None

  def print(self):
    self.print_header('Aggregate Scores')
    print(f'{self.scorer_name}:')
    for i, (score, string) in enumerate(zip(self.scores, self.strs)):
      if string is not None:
        print(f' Sys{i}: {score} ({string})')
      else:
        print(f' Sys{i}: {score}')
      
    if self.wins is not None:
      print('Significance test.')
      wins, sys_stats = self.wins, self.sys_stats
      for i, (win, (left, right)) in enumerate(zip(wins, self.compare_directions)):
        print(f'Win ratio: Sys{left}={win[0]:.3f}, Sys{right}={win[1]:.3f}, tie={win[2]:.3f}')
        if win[0] > win[1]:
          print(f'(Sys{left} is superior to Sys{right} with p value p={(1-win[0]):.3f})')
        elif win[1] > win[0]:
          print(f'(Sys{right} is superior to Sys{left} with p value p={(1-win[1]):.3f})')

      for i, sys_stat in enumerate(sys_stats):
        print(f'Sys{i}: mean={sys_stat["mean"]:.3f}, median={sys_stat["median"]:.3f}, 95%% confidence interval=[{sys_stat["lower_bound"]:.3f}, {sys_stat["upper_bound"]:.3f}]') 
      print()
    
  def plot(self, output_directory='outputs', output_fig_file='score', output_fig_format='pdf'):
    self.output_fig_file = output_fig_file
    if self.wins is not None:
      wins, sys_stats = self.wins, self.sys_stats
      mean, lrb, urb = [], [], []
      sys, sys_errs = [], []
      for i, sys_stat in enumerate(sys_stats):
        mean.append(sys_stat['mean'])
        lrb.append(sys_stat['lower_bound'])
        urb.append(sys_stat['upper_bound'])
        sys.append([self.scores[i], sys_stat['mean'], sys_stat['median']])
        N = len(sys[-1])
        sys_err = np.zeros((2, N))
        sys_err[0, 0] = self.scores[i] - lrb[-1] 
        sys_err[1, 0] = urb[-1] - self.scores[i]
        sys_errs.append(sys_err)
      xticklabels = [self.scorer_name, 'Bootstrap Mean', 'Bootstrap Median']
    else:
      sys = [[score] for score in self.scores]
      sys_errs = None
      xticklabels = None

    make_bar_chart(sys,
                   output_directory, output_fig_file, output_fig_format=output_fig_format,
                   errs=sys_errs, title='Score Comparison', ylabel=self.scorer_name,
                   xticklabels=xticklabels)
    
  def html_content(self, output_directory):
    line = []
    for i in range(len(self.scores)):
      line.append(f'Sys{i}')
    table = [line]
    for i, (score, string) in enumerate(zip(self.scores, self.strs)):
      if string is not None:
        line.append(f' Sys{i}: {score:.4f} ({string})')
      else:
        line.append(f' Sys{i}: {score:.4f}')
    table.append(line)
    html = html_table(table, caption=self.scorer_name)
    if self.wins is not None:
      wins, sys_stats = self.wins, self.sys_stats
      table = []
      name = ['Metric']
      mean = ['Mean']
      median = ['Median']
      interval = ['95% confidence interval']
      for i, sys_stat in enumerate(sys_stats):
        name.append(f'Sys{i}')
        mean.append(f'{sys_stat["mean"]:.3f}')
        median.append(f'{sys_stat["median"]:.3f}')
        interval.append(f'[{sys_stat["lower_bound"]:.3f},{sys_stat["upper_bound"]:.3f}]')
      table = [name, mean, median, interval]
      html += html_table(table, caption='Score Statistics') 
      table = []
      for i, (left, right) in enumerate(self.compare_directions):
        name = ['System names', f'Sys{left}', f'Sys{right}']
        win = ['Win ratio', f'{self.wins[i][0]:.3f}', f'{self.wins[i][1]:.3f}']
        table.append(name)
        table.append(win)
      html += html_table(table, caption='Significance Test', first_line_bold=False) 
    self.plot(output_directory, self.output_fig_file, 'png')
    html += html_img_reference(self.output_fig_file, 'Score Comparison')
    return html
    
class WordReport(Report):
  def __init__(self, bucketer, matches, acc_type, header):
    self.bucketer = bucketer
    self.matches = [[m for m in match] for match in matches]
    self.acc_type = acc_type
    self.header = header 
    self.acc_type_map = {'prec': 3, 'rec': 4, 'fmeas': 5}
    self.output_fig_file = None
    self.output_fig_format = None

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

  def plot(self, output_directory='outputs', output_fig_file='word-acc', output_fig_format='pdf'):
    width = 0.35
    self.output_fig_file = output_fig_file
    self.output_fig_format = output_fig_format
    acc_types = self.acc_type.split('+')
    for at in acc_types:
      if at not in self.acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = self.acc_type_map[at]
      sys = [[m[aid] for m in match] for match in self.matches]
      xticklabels = [s for s in self.bucketer.bucket_strs] 

      make_bar_chart(sys,
                     output_directory, output_fig_file, output_fig_format=output_fig_format,
                     title='Word Accuracy Comparison', ylabel=at,
                     xticklabels=xticklabels)
    
  def html_content(self, output_directory):
    acc_type_map = self.acc_type_map
    bucketer, matches, acc_type, header = self.bucketer, self.matches, self.acc_type, self.header
    acc_types = acc_type.split('+') 
    
    self.plot(output_directory, self.output_fig_file, 'png')

    html = ''
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      caption = f'Word {acc_type} by {bucketer.name()} bucket'
      table = [[bucketer.name()]]
      for i in range(len(matches)):
        table[0].append(f'Sys{i}')
      for i, bs in enumerate(bucketer.bucket_strs):
        line = [bs]
        for match in matches:
          line.append(f'{match[i][aid]:.4f}')
        table += [line] 
      html += html_table(table, caption)
      html += html_img_reference(f'{self.output_fig_file}-{at}', self.header)
    return html 

class NgramReport(Report):
  def __init__(self, scorelist, report_length, min_ngram_length, max_ngram_length, matches, compare_type, alpha, compare_directions=[(0, 1)], label_files=None):
    self.scorelist = scorelist
    self.report_length = report_length 
    self.min_ngram_length = min_ngram_length
    self.max_ngram_length = max_ngram_length
    self.matches = matches
    self.compare_type = compare_type 
    self.label_files = label_files
    self.alpha = alpha
    self.compare_directions = compare_directions
    self.output_fig_file = None

  def print(self):
    report_length = self.report_length
    self.print_header('N-gram Difference Analysis')
    print(f'--- min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
    print(f'    report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')
    if self.label_files is not None:
      print(self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      print(f'--- {report_length} n-grams that System {left} had higher {self.compare_type}')
      for k, v in self.scorelist[i][:report_length]:
        print('{}\t{} (sys{}={}, sys{}={})'.format(' '.join(k), v, left, self.matches[left][k], right, self.matches[right][k]))
      print(f'\n--- {report_length} n-grams that System {right} had higher {self.compare_type}')
      for k, v in reversed(self.scorelist[i][-report_length:]):
        print('{}\t{} (sys{}={}, sys{}={})'.format(' '.join(k), v, left, self.matches[left][k], right, self.matches[right][k]))
      print()

  def plot(self, output_directory='outputs', output_fig_file='score', output_fig_format='pdf'):
    pass 

  def html_content(self, output_directory=None):
    report_length = self.report_length
    html = tag_str('p', f'min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
    html += tag_str('p', f'report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')
    if self.label_files is not None:
      html += tag_str('p', self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      caption = f'{report_length} n-grams that System {left} had higher {self.compare_type}'
      table = [['n-gram', self.compare_type, f'Sys{left}', f'Sys{right}']]
      table.extend([[' '.join(k), f'{v:.2f}', self.matches[left][k], self.matches[right][k]] for k, v in self.scorelist[i][:report_length]])
      html += html_table(table, caption)

      caption = f'{report_length} n-grams that System {right} had higher {self.compare_type}'
      table = [['n-gram', self.compare_type, f'Sys{left}', f'Sys{right}']]
      table.extend([[' '.join(k), f'{v:.2f}', self.matches[left][k], self.matches[right][k]] for k, v in reversed(self.scorelist[i][-report_length:])])
      html += html_table(table, caption)
    return html 

class SentenceReport(Report):
  def __init__(self, bucketer=None, bucket_type=None, sys_stats=None, statistic_type=None, score_measure=None):
    self.bucketer = bucketer
    self.bucket_type = bucket_type 
    self.sys_stats = [[s for s in stat] for stat in sys_stats]
    self.statistic_type = statistic_type
    self.score_measure = score_measure
    self.output_fig_file = None

  def print(self):
    bucketer, stats, bucket_type, statistic_type, score_measure = self.bucketer, self.sys_stats, self.bucket_type, self.statistic_type, self.score_measure
    self.print_header('Sentence Bucket Analysis')
    print(f'--- bucket_type={bucket_type}, statistic_type={statistic_type}, score_measure={score_measure}')
    for i, bs in enumerate(bucketer.bucket_strs):
      print(f'{bs}', end='')
      for stat in stats:
        print(f'\t{stat[i]}', end='')
      print()
    print()

  def plot(self, output_directory='outputs', output_fig_file='word-acc', output_fig_format='pdf'):
    self.output_fig_file = output_fig_file
    sys = self.sys_stats
    xticklabels = [s for s in self.bucketer.bucket_strs] 

    make_bar_chart(sys,
                   output_directory, output_fig_file, output_fig_format=output_fig_format,
                   title='Sentence Bucket Analysis', xlabel=self.bucket_type, ylabel=self.statistic_type,
                   xticklabels=xticklabels)

  def html_content(self, output_directory=None):
    bucketer, stats, bucket_type, statistic_type, score_measure = self.bucketer, self.sys_stats, self.bucket_type, self.statistic_type, self.score_measure
    caption = (f'bucket_type={bucket_type}, statistic_type={statistic_type}, score_measure={score_measure}')
    table = [[bucket_type]]
    for i in range(len(stats)):
      table[0].append(f'Sys{i}')
    for i, bs in enumerate(bucketer.bucket_strs):
      line = [bs]
      for stat in stats:
        line.append(f'{stat[i]:.4f}')
      table.extend([line])
    html = html_table(table, caption)
    self.plot(output_directory, self.output_fig_file, 'png')
    html += html_img_reference(self.output_fig_file, 'Sentence Bucket Analysis')
    return html 

class SentenceExampleReport(Report):
  def __init__(self, report_length=None, scorediff_lists=None, scorer_name=None, ref=None, outs=None, compare_directions=[(0, 1)]):
    self.report_length = report_length 
    self.scorediff_lists = scorediff_lists
    self.scorer_name = scorer_name
    self.ref = ref 
    self.outs = outs
    self.compare_directions = compare_directions

  def print(self):
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      print(f'--- {report_length} sentences where Sys{left}>Sys{right} at {self.scorer_name}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        print ('sys{}-sys{}={}, sys{}={}, sys{}={}\nRef:  {}\nSys{}: {}\nSys{}: {}\n'.format(left, right, -bdiff, left, s1, right, s2, ' '.join(ref[i]), left, ' '.join(out1[i]), right, ' '.join(out2[i])))
      print(f'--- {report_length} sentences where Sys{right}>Sys{left} at {self.scorer_name}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        print ('sys{}-sys{}={}, sys{}={}, sys{}={}\nRef:  {}\nSys{}: {}\nSys{}: {}\n'.format(right, left, bdiff, left, s1, right, s2, ' '.join(ref[i]), left, ' '.join(out1[i]), right, ' '.join(out2[i])))


  def plot(self, output_directory='outputs', output_fig_file='word-acc', output_fig_format='pdf'):
    pass 

  def html_content(self, output_directory=None):
    report_length = self.report_length 
    for cnt, (left, right) in enumerate(self.compare_directions):
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      html = tag_str('h4', f'{report_length} sentences where Sys{left}>Sys{right} at {self.scorer_name}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        caption = f'sys{left}-sys{right}={-bdiff:.2f}, sys{left}={s1:.2f}, sys{right}={s2:.2f}'
        table = [['Ref', ' '.join(ref[i])], [f'Sys{left}', ' '.join(out1[i])], [f'Sys{right}', ' '.join(out2[i])]]
        html += html_table(table, caption, first_line_bold=False)

      html += tag_str('h4', f'{report_length} sentences where Sys{left}>Sys{right} at {self.scorer_name}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        caption = f'sys{right}-sys{left}={bdiff:.2f}, sys{left}={s1:.2f}, sys{right}={s2:.2f}'
        table = [['Ref', ' '.join(ref[i])], [f'Sys{left}', ' '.join(out1[i])], [f'Sys{right}', ' '.join(out2[i])]]
        html += html_table(table, caption, first_line_bold=False)

    return html 


def tag_str(tag, str, new_line=''):
  return f'<{tag}>{new_line} {str} {new_line}</{tag}>'

def html_table(table, caption=None, first_line_bold=True):
  html = '<table border="1">\n'
  if caption is not None:
    html += tag_str('caption', caption)
  if first_line_bold:
    html += '\n  '.join(tag_str('th', ri) for ri in table[0])
  else:
    html += '\n  '.join(tag_str('td', ri) for ri in table[0])
  for row in table[1:]:
    table_row = '\n  '.join(tag_str('td', ri) for ri in row)
    html += tag_str('tr', table_row)
  html += '\n</table>\n <br>'
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
  
