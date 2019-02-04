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

plot_counter = 0
def next_plot_id():
  global plot_counter
  plot_counter += 1
  return f'{plot_counter:03d}'

bar_colors = ["#7293CB", "#E1974C", "#84BA5B", "#D35E60", "#808585", "#9067A7", "#AB6857", "#CCC210"]

def make_bar_chart(datas,
                   output_directory, output_fig_file, sys_names, output_fig_format='png',
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

  def generate_report(self, output_fig_file=None, output_fig_format=None, output_directory=None):
    self.print()

class ScoreReport(Report):
  def __init__(self, scorer, scores, strs, sys_names,
               wins=None, sys_stats=None, compare_directions=None):
    self.scorer = scorer 
    self.scores = scores
    self.strs = strs
    self.wins = wins
    self.sys_stats = sys_stats
    self.sys_names = sys_names
    self.compare_directions = compare_directions
    self.output_fig_file = f'{next_plot_id()}-score-{scorer.idstr()}'

  def print(self):
    self.print_header('Aggregate Scores')
    print(f'{self.scorer.name()}:')
    for i, (score, string) in enumerate(zip(self.scores, self.strs)):
      if string is not None:
        print(f' {self.sys_names[i]}: {score} ({string})')
      else:
        print(f' {self.sys_names[i]}: {score}')
      
    if self.wins is not None:
      print('Significance test.')
      wins, sys_stats = self.wins, self.sys_stats
      for i, (win, (left, right)) in enumerate(zip(wins, self.compare_directions)):
        print(f'Win ratio: {self.sys_names[left]}={win[0]:.3f}, {self.sys_names[right]}={win[1]:.3f}, tie={win[2]:.3f}')
        if win[0] > win[1]:
          print(f'({self.sys_names[left]} is superior to {self.sys_names[right]} with p value p={(1-win[0]):.3f})')
        elif win[1] > win[0]:
          print(f'({self.sys_names[right]} is superior to {self.sys_names[left]} with p value p={(1-win[1]):.3f})')

      for i, (sys_stat, sys_name) in enumerate(zip(sys_stats, sys_names)):
        print(f'{sys_name}: mean={sys_stat["mean"]:.3f}, median={sys_stat["median"]:.3f}, 95%% confidence interval=[{sys_stat["lower_bound"]:.3f}, {sys_stat["upper_bound"]:.3f}]')
      print()
    
  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
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
      xticklabels = [self.scorer.name(), 'Bootstrap Mean', 'Bootstrap Median']
    else:
      sys = [[score] for score in self.scores]
      sys_errs = None
      xticklabels = None

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   sys_names=self.sys_names,
                   output_fig_format=output_fig_format,
                   errs=sys_errs, title='Score Comparison', ylabel=self.scorer.name(),
                   xticklabels=xticklabels)

  def html_content(self, output_directory):
    table = [self.sys_names]
    line = []
    for i, (score, string) in enumerate(zip(self.scores, self.strs)):
      if string is not None:
        line.append(f'{score:.4f} ({string})')
      else:
        line.append(f'{score:.4f}')
    table.append(line)

    html = html_table(table, caption=self.scorer.name())

    if self.wins is not None:
      wins, sys_stats = self.wins, self.sys_stats
      table = []
      name = ['Metric'] + self.sys_names
      mean = ['Mean']
      median = ['Median']
      interval = ['95% confidence interval']
      for i, sys_stat in enumerate(sys_stats):
        mean.append(f'{sys_stat["mean"]:.3f}')
        median.append(f'{sys_stat["median"]:.3f}')
        interval.append(f'[{sys_stat["lower_bound"]:.3f},{sys_stat["upper_bound"]:.3f}]')
      table = [name, mean, median, interval]
      html += html_table(table, caption='Score Statistics') 
      table = []
      for i, (left, right) in enumerate(self.compare_directions):
        name = ['System names', f'{self.sys_names[left]}', f'{self.sys_names[right]}']
        win = ['Win ratio', f'{self.wins[i][0]:.3f}', f'{self.wins[i][1]:.3f}']
        table.append(name)
        table.append(win)
      html += html_table(table, caption='Significance Test', first_line_bold=False) 

    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Score Comparison')
    return html
    
class WordReport(Report):
  def __init__(self, bucketer, matches, acc_type, header, sys_names):
    self.bucketer = bucketer
    self.matches = [[m for m in match] for match in matches]
    self.acc_type = acc_type
    self.header = header
    self.sys_names = sys_names
    self.acc_type_map = {'prec': 3, 'rec': 4, 'fmeas': 5}
    self.output_fig_file = f'{next_plot_id()}-wordacc-{bucketer.name()}'

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
                     sys_names=self.sys_names,
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
      table = [[bucketer.name()] + self.sys_names]
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
               matches, compare_type, alpha, sys_names, compare_directions=[(0, 1)], label_files=None):
    self.scorelist = scorelist
    self.report_length = report_length 
    self.min_ngram_length = min_ngram_length
    self.max_ngram_length = max_ngram_length
    self.matches = matches
    self.compare_type = compare_type
    self.sys_names = sys_names
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
      print(f'--- {report_length} n-grams that {self.sys_names[left]} had higher {self.compare_type}')
      for k, v in self.scorelist[i][:report_length]:
        print('{}\t{} (sys{}={}, sys{}={})'.format(' '.join(k), v, left+1, self.matches[left][k], right+1, self.matches[right][k]))
      print(f'\n--- {report_length} n-grams that {self.sys_names[right]} had higher {self.compare_type}')
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
      caption = f'{report_length} n-grams that {self.sys_names[left]} had higher {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{self.sys_names[left]}', f'{self.sys_names[right]}']]
      table.extend([[' '.join(k), f'{v:.2f}', self.matches[left][k], self.matches[right][k]] for k, v in self.scorelist[i][:report_length]])
      html += html_table(table, caption)

      caption = f'{report_length} n-grams that {self.sys_names[right]} had higher {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{self.sys_names[left]}', f'{self.sys_names[right]}']]
      table.extend([[' '.join(k), f'{v:.2f}', self.matches[left][k], self.matches[right][k]] for k, v in reversed(self.scorelist[i][-report_length:])])
      html += html_table(table, caption)
    return html 

class SentenceReport(Report):

  def __init__(self, sys_names, bucketer=None, sys_stats=None, statistic_type=None, scorer=None):
    self.bucketer = bucketer
    self.sys_stats = [[s for s in stat] for stat in sys_stats]
    self.sys_names = sys_names
    self.statistic_type = statistic_type
    self.scorer = scorer
    self.yname = scorer.name() if statistic_type == 'score' else statistic_type
    self.yidstr = scorer.idstr() if statistic_type == 'score' else statistic_type
    self.output_fig_file = f'{next_plot_id()}-sent-{bucketer.idstr()}-{self.yidstr}'
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
                   sys_names=self.sys_names,
                   output_fig_format=output_fig_format,
                   title='Sentence Bucket Analysis', xlabel=self.bucketer.name(), ylabel=self.yname,
                   xticklabels=xticklabels)

  def html_content(self, output_directory=None):
    table = [ [self.bucketer.name()] + self.sys_names ]
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

  def __init__(self, sys_names, report_length=None, scorediff_lists=None, scorer=None, ref=None, outs=None, compare_directions=[(0, 1)]):
    self.report_length = report_length 
    self.scorediff_lists = scorediff_lists
    self.scorer = scorer
    self.sys_names = sys_names
    self.ref = ref 
    self.outs = outs
    self.compare_directions = compare_directions

  def print(self):
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      sleft, sright = self.sys_names[left], self.sys_names[right]
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
      sleft, sright = self.sys_names[left], self.sys_names[right]
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      html = tag_str('h4', f'{report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        caption = f'{sleft}-{sright}={-bdiff:.2f}, {sleft}={s1:.2f}, {sright}={s2:.2f}'
        table = [['Ref', ' '.join(ref[i])], [f'{sleft}', ' '.join(out1[i])], [f'{sright}', ' '.join(out2[i])]]
        html += html_table(table, caption, first_line_bold=False)

      html += tag_str('h4', f'{report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        caption = f'{sright}-{sleft}={bdiff:.2f}, {sleft}={s1:.2f}, {sright}={s2:.2f}'
        table = [['Ref', ' '.join(ref[i])], [f'{sleft}', ' '.join(out1[i])], [f'{sright}', ' '.join(out2[i])]]
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
  
