# Overall imports
import argparse
import operator
import itertools
import nltk
from collections import defaultdict

# In-package imports
import count_ngrams
import stat_utils
import scorers


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

def load_tokens(filename):
  with open(filename, "r") as f:
    return [line.strip().split() for line in f]

def parse_profile(profile):
  kargs = {}
  for kv in profile.split(','):
    k, v = kv.split('=')
    kargs[k] = v
  return kargs

def print_score_report(ref, out1, out2,
                       score_type='bleu'):
  """
  Print a report comparing overall scores of the two systems.

  Args:
    ref: Tokens from the reference
    out1: Tokens from the output file 1
    out2: Tokens from the output file 2
    score_type: A string specifying the scoring type (bleu/length)
  """
  scorer = scorers.create_scorer_from_profile(score_type)
  print(f'{scorer.name()}:')
  score1, str1 = scorer.score_corpus(ref,out1)
  score2, str2 = scorer.score_corpus(ref,out2)
  if str1 is not None:
    print(f' Sys1: {score1} ({str1})\n Sys2: {score2} ({str2})')
  else:
    print(f' Sys1: {score1}\n Sys2: {score2}')

def print_ngram_report(ref, out1, out2,
                       min_ngram_length=1, max_ngram_length=4,
                       report_length=50, alpha=1.0, compare_type='match',
                       ref_labels=None, out1_labels=None, out2_labels=None):
  """
  Print a report comparing aggregate n-gram statistics

  Args:
    ref: Tokens from the reference
    out1: Tokens from the output file 1
    out2: Tokens from the output file 2
    min_ngram_length: minimum n-gram length
    max_ngram_length: maximum n-gram length
    report_length: the number of n-grams to report
    alpha: when sorting n-grams for salient features, the smoothing coefficient
    compare_type: what type of statistic to compare
                  (match: n-grams that match the reference, over: over-produced ngrams, under: under-produced ngrams)
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`.
                If specified, will aggregate statistics over labels instead of n-grams.
    out1_labels: output 1 labels. must be specified if ref_labels is specified.
    out2_labels: output 2 labels. must be specified if ref_labels is specified.
  """
  print(f'params: min_ngram_length={min_ngram_length}, max_ngram_length={max_ngram_length}')
  print(f'        report_length={report_length}, alpha={alpha}, compare_type={compare_type}')
  if type(ref_labels) == str:
    print(f'        ref_labels={ref_labels}, out1_labels={out1_labels}, out2_labels={out2_labels}')

  ref_labels = load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
  out1_labels = load_tokens(out1_labels) if type(out1_labels) == str else out1_labels
  out2_labels = load_tokens(out2_labels) if type(out2_labels) == str else out2_labels
  total1, match1, over1, under1 = count_ngrams.compare_ngrams(ref, out1, ref_labels=ref_labels, out_labels=out1_labels,
                                                              min_length=min_ngram_length, max_length=max_ngram_length)
  total2, match2, over2, under2 = count_ngrams.compare_ngrams(ref, out2, ref_labels=ref_labels, out_labels=out2_labels,
                                                              min_length=min_ngram_length, max_length=max_ngram_length)
  if compare_type == 'match':
    scores = stat_utils.extract_salient_features(match1, match2, alpha=1)
  elif compare_type == 'over':
    scores = stat_utils.extract_salient_features(over1, over2, alpha=1)
  elif compare_type == 'under':
    scores = stat_utils.extract_salient_features(under1, under2, alpha=1)
  else:
    raise ValueError(f'Illegal compare_type "{compare_type}"')
  scorelist = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

  print(f'--- {report_length} n-grams that System 1 had higher {compare_type}')
  for k, v in scorelist[:report_length]:
    print('{}\t{} (sys1={}, sys2={})'.format(' '.join(k), v, match1[k], match2[k]))
  print(f'\n--- {report_length} n-grams that System 2 had higher {compare_type}')
  for k, v in reversed(scorelist[-report_length:]):
    print('{}\t{} (sys1={}, sys2={})'.format(' '.join(k), v, match1[k], match2[k]))

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      description='Program to compare MT results',
  )
  parser.add_argument('ref_file', type=str,
                      help='A path to a correct reference file')
  parser.add_argument('out1_file', type=str,
                      help='A path to a system output')
  parser.add_argument('out2_file', type=str,
                      help='A path to another system output')
  parser.add_argument('--compare_scores', type=str, default=['score_type=bleu', 'score_type=length'], nargs='*',
                      help="""
                      Compare scores. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'print_score_report' to see which arguments are available.
                      """)
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

  if args.compare_scores:
    print('********************** Aggregate Scores ************************')
    for profile in args.compare_scores:
      kargs = parse_profile(profile)
      print_score_report(ref, out1, out2, **kargs)
      print()

  if args.compare_ngrams:
    for profile in args.compare_ngrams:
      kargs = parse_profile(profile)
      print('********************** N-gram Difference Analysis ************************')
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
