# Overall imports
import sys
import argparse
import operator
import itertools
import nltk
from collections import defaultdict

# In-package imports
import count_ngrams
import stat_utils

parser = argparse.ArgumentParser(
    description='Program to compare MT results',
)
parser.add_argument('ref_file', type=str,
                    help='A path to a correct reference file')
parser.add_argument('out1_file', type=str,
                    help='A path to a system output')
parser.add_argument('out2_file', type=str,
                    help='A path to another system output')
parser.add_argument('--compare_ngrams', type=str, default=['compare_type=match'], nargs='*',
                    help="""
                    Compare ngrams. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                    See documentation for 'print_ngram_report' to see which arguments are available.
                    """)
parser.add_argument('--train_file', type=str, default=None,
                    help='A link to the training corpus target file')
parser.add_argument('--train_counts', type=str, default=None,
                    help='A link to the training word frequency counts as a tab-separated "word\\tfreq" file')
parser.add_argument('--sent_size', type=int, default=10,
                    help='How many sentences to print.')
args = parser.parse_args()

def load_tokens(filename):
  with open(filename, "r") as f:
    return [line.strip().split() for line in f]

ref, out1, out2 = [load_tokens(x) for x in (args.ref_file, args.out1_file, args.out2_file)]

# Calculate the frequency counts, from the training corpus
# or training frequency file if either are specified, from the
# reference file if not
freq_counts = defaultdict(lambda: 0)
if args.train_counts != None:
  with open(args.train_counts, "r") as f:
    for line in f:
      word, freq = line.strip().split('\t')
      freq_counts[word] = freq
else:
  my_file = args.train_file if args.train_file != None else args.ref_file
  with open(my_file, "r") as f:
    for line in f:
      for word in line.strip().split():
        freq_counts[word] += 1 

def calc_matches_by_freq(ref, out, buckets):
  extended_buckets = buckets + [max(freq_counts.values()) + 1]
  matches = [[0,0,0] for x in extended_buckets]
  for refsent, outsent in zip(ref, out):
    reffreq, outfreq = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for x in refsent:
      reffreq[x] += 1
    for x in outsent:
      outfreq[x] += 1
    for k in set(itertools.chain(reffreq.keys(), outfreq.keys())):
      for bucket, match in zip(extended_buckets, matches):
        if freq_counts[k] < bucket:
          match[0] += min(reffreq[k], outfreq[k])
          match[1] += reffreq[k]
          match[2] += outfreq[k]
          break 
  for bothf, reff, outf in matches:
    if bothf == 0:
      rec, prec, fmeas = 0.0, 0.0, 0.0
    else:
      rec = bothf / float(reff)
      prec = bothf / float(outf)
      fmeas = 2 * prec * rec / (prec + rec)
    yield bothf, reff, outf, rec, prec, fmeas

buckets = [1, 2, 3, 4, 5, 10, 100, 1000]
bucket_strs = []
last_start = 0
for x in buckets:
  if x-1 == last_start:
    bucket_strs.append(str(last_start))
  else:
    bucket_strs.append("{}-{}".format(last_start, x-1))
  last_start = x
bucket_strs.append("{}+".format(last_start))

def print_ngram_report(ref, out1, out2,
                       max_ngram_length=4, report_length=50, alpha=1.0, compare_type='match',
                       ref_label_file=None, out1_label_file=None, out2_label_file=None):
  """
  Print a report comparing aggregate n-gram statistics

  Args:
    ref: Tokens from the reference
    out1: Tokens from the output file 1
    out2: Tokens from the output file 2
    max_ngram_length: maximum n-gram length
    report_length: the number of n-grams to report
    alpha: when sorting n-grams for salient features, the smoothing coefficient
    compare_type: what type of statistic to compare (match: n-grams that match the reference, over: over-produced ngrams, under: under-produced ngrams)
    ref_label_file: reference label file (one label for each word).
                will aggregate statistics over labels instead of n-grams.
    out1_label_file: output 1 label file. must be specified if ref_labels is specified.
    out2_label_file: output 2 label file. must be specified if ref_labels is specified.
  """
  ref_labels = load_tokens(ref_label_file) if ref_label_file else None
  out1_labels = load_tokens(out1_label_file) if out1_label_file else None
  out2_labels = load_tokens(out2_label_file) if out2_label_file else None
  total1, match1, over1, under1 = count_ngrams.compare_ngrams(ref, out1, ref_labels=ref_labels, out_labels=out1_labels, max_length=max_ngram_length)
  total2, match2, over2, under2 = count_ngrams.compare_ngrams(ref, out2, ref_labels=ref_labels, out_labels=out2_labels, max_length=max_ngram_length)
  if compare_type == 'match':
    scores = stat_utils.extract_salient_features(match1, match2, alpha=1)
  elif compare_type == 'over':
    scores = stat_utils.extract_salient_features(over1, over2, alpha=1)
  elif compare_type == 'under':
    scores = stat_utils.extract_salient_features(under1, under2, alpha=1)
  else:
    raise ValueError(f'Illegal compare_type "{compare_type}"')
  scorelist = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
  print('********************** N-gram Difference Analysis ************************')
  print(f'params: max_ngram_length={max_ngram_length}, report_length={report_length}, alpha={alpha}, compare_type={compare_type}')
  if ref_label_file:
    print(f'        ref_label_file={ref_label_file}, out1_label_file={out1_label_file}, out2_label_file={out2_label_file}')

  print(f'--- {report_length} n-grams that System 1 had higher {compare_type}')
  for k, v in scorelist[:report_length]:
    print('{}\t{} (sys1={}, sys2={})'.format(' '.join(k), v, match1[k], match2[k]))
  print(f'\n--- {report_length} n-grams that System 2 had higher {compare_type}')
  for k, v in reversed(scorelist[-report_length:]):
    print('{}\t{} (sys1={}, sys2={})'.format(' '.join(k), v, match1[k], match2[k]))

if args.compare_ngrams:
  for profile in args.compare_ngrams:
    kargs = {}
    for kv in profile.split(','):
      k, v = kv.split('=')
      kargs[k] = v
    print_ngram_report(ref, out1, out2, **kargs)

# Calculate f-measure
matches = calc_matches_by_freq(ref, out1, buckets)
matches2 = calc_matches_by_freq(ref, out2, buckets)
print('\n\n********************** Word Frequency Analysis ************************')
print('--- word f-measure by frequency bucket')
for bucket_str, match, match2 in zip(bucket_strs, matches, matches2):
  print("{}\t{:.4f}\t{:.4f}".format(bucket_str, match[5], match2[5]))
# Calculate BLEU diff
scorediff_list = []
chencherry = nltk.translate.bleu_score.SmoothingFunction()
for i, (o1, o2, r) in enumerate(zip(out1, out2, ref)):
  b1 = nltk.translate.bleu_score.sentence_bleu([r], o1, smoothing_function=chencherry.method2)
  b2 = nltk.translate.bleu_score.sentence_bleu([r], o2, smoothing_function=chencherry.method2)
  scorediff_list.append((b2-b1, b1, b2, i))
scorediff_list.sort()
print('\n\n********************** Length Analysis ************************')
print('--- length ratio')
length_ref = sum([len(x) for x in ref])
length_out1 = sum([len(x) for x in out1])
length_out2 = sum([len(x) for x in out2])
print('System 1: {}, System 2: {}'.format(length_out1/length_ref, length_out2/length_ref))
print('--- length difference from reference by bucket')
length_diff = {}
length_diff2 = {}
for r, o, o2 in zip(ref, out1, out2):
  ld = len(o)-len(r)
  ld2 = len(o2)-len(r)
  length_diff[ld] = length_diff.get(ld,0) + 1
  length_diff2[ld2] = length_diff2.get(ld2,0) + 1
for ld in sorted(list(set(length_diff.keys()) | set(length_diff2.keys()))):
  print("{}\t{}\t{}".format(ld, length_diff.get(ld,0), length_diff2.get(ld,0)))
print('\n\n********************** BLEU Analysis ************************')
print('--- %d sentences that System 1 did a better job at than System 2' % args.sent_size)
for bdiff, b1, b2, i in scorediff_list[:args.sent_size]:
  print ('BLEU+1 sys2-sys1={}, sys1={}, sys2={}\nRef:  {}\nSys1: {}\nSys2: {}\n'.format(bdiff, b1, b2, ' '.join(ref[i]), ' '.join(out1[i]), ' '.join(out2[i])))
print('--- %d sentences that System 2 did a better job at than System 1' % args.sent_size)
for bdiff, b1, b2, i in scorediff_list[-args.sent_size:]:
  print ('BLEU+1 sys2-sys1={}, sys1={}, sys2={}\nRef:  {}\nSys1: {}\nSys2: {}\n'.format(bdiff, b1, b2, ' '.join(ref[i]), ' '.join(out1[i]), ' '.join(out2[i])))
