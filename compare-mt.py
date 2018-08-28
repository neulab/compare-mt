import sys
import argparse
import operator
import itertools
import nltk
from collections import defaultdict

parser = argparse.ArgumentParser(
    description='Program to compare MT results',
)
parser.add_argument('ref_file', type=str, help='A path to a correct reference file')
parser.add_argument('out_file', type=str, help='A path to a system output')
parser.add_argument('out2_file', nargs='?', type=str, default=None, help='A path to another system output. Add only if you want to compare outputs from two systems.')
parser.add_argument('--train_file', type=str, default=None, help='A link to the training corpus target file')
parser.add_argument('--train_counts', type=str, default=None, help='A link to the training word frequency counts as a tab-separated "word\\tfreq" file')
parser.add_argument('--alpha', type=float, default=1.0, help='A smoothing coefficient to control how much the model focuses on low- and high-frequency events. 1.0 should be fine most of the time.')
parser.add_argument('--ngram', type=int, default=4, help='Maximum length of n-grams.')
parser.add_argument('--ngram_size', type=int, default=50, help='How many n-grams to print.')
parser.add_argument('--sent_size', type=int, default=10, help='How many sentences to print.')
args = parser.parse_args()

with open(args.ref_file, "r") as f:
  ref = [line.strip().split() for line in f]
with open(args.out_file, "r") as f:
  out = [line.strip().split() for line in f]
if args.out2_file != None:
  with open(args.out2_file, "r") as f:
    out2 = [line.strip().split() for line in f]

def calc_ngrams(sent):
  ret = defaultdict(lambda: 0)
  for n in range(args.ngram):
    for i in range(len(sent)-n):
      ret[tuple(sent[i:i+n+1])] += 1
  return ret

def match_ngrams(left, right):
  ret = defaultdict(lambda: 0)
  for k, v in left.items():
    if k in right:
      ret[k] = min(v, right[k])
  return ret

# Calculate over and under-generated n-grams for two corpora
def calc_over_under(ref, out, alpha):
  # Create n-grams
  refall = defaultdict(lambda: 0)
  outall = defaultdict(lambda: 0)
  for refsent, outsent in zip(ref, out):
    for k, v in calc_ngrams(refsent).items():
      refall[k] += v
    for k, v in calc_ngrams(outsent).items():
      outall[k] += v
  # Calculate scores
  scores = {}
  for k, v in refall.items():
    scores[k] = (v + args.alpha) / (v + outall[k] + 2*args.alpha)
  for k, v in outall.items():
    scores[k] = (refall[k] + args.alpha) / (refall[k] + v + 2*args.alpha)
  return refall, outall, scores

# Calculate over and under-generated n-grams for two corpora
def calc_compare(ref, out, out2, alpha):
  outall = defaultdict(lambda: 0)
  out2all = defaultdict(lambda: 0)
  for refsent, outsent, out2sent in zip(ref, out, out2):
    refn = calc_ngrams(refsent)
    outmatch = match_ngrams(refn, calc_ngrams(outsent))
    out2match = match_ngrams(refn, calc_ngrams(out2sent))
    for k, v in outmatch.items():
      if v > out2match[k]:
        outall[k] += v - out2match[k]
    for k, v in out2match.items():
      if v > outmatch[k]:
        out2all[k] += v - outmatch[k]
  # Calculate scores
  scores = {}
  for k, v in out2all.items():
    scores[k] = (v + args.alpha) / (v + outall[k] + 2*args.alpha)
  for k, v in outall.items():
    scores[k] = (out2all[k] + args.alpha) / (out2all[k] + v + 2*args.alpha)
  return outall, out2all, scores

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

# Analyze the reference/output
if args.out2_file == None:
  refall, outall, scores = calc_over_under(ref, out, args.alpha)
  scorelist = sorted(scores.items(), key=operator.itemgetter(1))
  # Print the ouput
  print('********************** N-gram Difference Analysis ************************')
  print('--- %d over-generated n-grams indicative of output' % args.ngram_size)
  for k, v in scorelist[:args.ngram_size]:
    print('%s\t%f (ref=%d, out=%d)' % (' '.join(k), v, refall[k], outall[k]))
  print()
  print('--- %d under-generated n-grams indicative of reference' % args.ngram_size)
  for k, v in reversed(scorelist[-args.ngram_size:]):
    print('%s\t%f (ref=%d, out=%d)' % (' '.join(k), v, refall[k], outall[k]))
  # Calculate f-measure
  matches = calc_matches_by_freq(ref, out, buckets)
  print('\n\n********************** Word Frequency Analysis ************************')
  print('--- word f-measure by frequency bucket')
  for bucket_str, match in zip(bucket_strs, matches):
    print("{}\t{:.4f}".format(bucket_str, match[5]))
# Analyze the differences between two systems
else:
  outall, out2all, scores = calc_compare(ref, out, out2, args.alpha)
  scorelist = sorted(scores.items(), key=operator.itemgetter(1))
  # Print the ouput
  print('********************** N-gram Difference Analysis ************************')
  print('--- %d n-grams that System 1 did a better job of producing' % args.ngram_size)
  for k, v in scorelist[:args.ngram_size]:
    print('%s\t%f (sys1=%d, sys2=%d)' % (' '.join(k), v, outall[k], out2all[k]))
  print('\n--- %d n-grams that System 2 did a better job of producing' % args.ngram_size)
  for k, v in reversed(scorelist[-args.ngram_size:]):
    print('%s\t%f (sys1=%d, sys2=%d)' % (' '.join(k), v, outall[k], out2all[k]))  
  # Calculate f-measure
  matches = calc_matches_by_freq(ref, out, buckets)
  matches2 = calc_matches_by_freq(ref, out2, buckets)
  print('\n\n********************** Word Frequency Analysis ************************')
  print('--- word f-measure by frequency bucket')
  for bucket_str, match, match2 in zip(bucket_strs, matches, matches2):
    print("{}\t{:.4f}\t{:.4f}".format(bucket_str, match[5], match2[5]))
  # Calculate BLEU diff
  scorediff_list = []
  chencherry = nltk.translate.bleu_score.SmoothingFunction()
  for i, (o1, o2, r) in enumerate(zip(out, out2, ref)):
    b1 = nltk.translate.bleu_score.sentence_bleu([r], o1, smoothing_function=chencherry.method2)
    b2 = nltk.translate.bleu_score.sentence_bleu([r], o2, smoothing_function=chencherry.method2)
    scorediff_list.append((b2-b1, b1, b2, i))
  scorediff_list.sort()
  print('\n\n********************** Length Analysis ************************')
  print('--- length ratio')
  length_ref = sum([len(x) for x in ref])
  length_out = sum([len(x) for x in out])
  length_out2 = sum([len(x) for x in out2])
  print('System 1: {}, System 2: {}'.format(length_out/length_ref, length_out2/length_ref))
  print('--- length difference from reference by bucket')
  length_diff = {}
  length_diff2 = {}
  for r, o, o2 in zip(ref, out, out2):
    ld = len(o)-len(r)
    ld2 = len(o2)-len(r)
    length_diff[ld] = length_diff.get(ld,0) + 1
    length_diff2[ld2] = length_diff2.get(ld2,0) + 1
  for ld in sorted(list(set(length_diff.keys()) | set(length_diff2.keys()))):
    print("{}\t{}\t{}".format(ld, length_diff.get(ld,0), length_diff2.get(ld,0)))
  print('\n\n********************** BLEU Analysis ************************')
  print('--- %d sentences that System 1 did a better job at than System 2' % args.sent_size)
  for bdiff, b1, b2, i in scorediff_list[:args.sent_size]:
    print ('BLEU+1 sys2-sys1={}, sys1={}, sys2={}\nRef:  {}\nSys1: {}\nSys2: {}\n'.format(bdiff, b1, b2, ' '.join(ref[i]), ' '.join(out[i]), ' '.join(out2[i])))
  print('--- %d sentences that System 2 did a better job at than System 1' % args.sent_size)
  for bdiff, b1, b2, i in scorediff_list[-args.sent_size:]:
    print ('BLEU+1 sys2-sys1={}, sys1={}, sys2={}\nRef:  {}\nSys1: {}\nSys2: {}\n'.format(bdiff, b1, b2, ' '.join(ref[i]), ' '.join(out[i]), ' '.join(out2[i])))
