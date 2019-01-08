# Overall imports
import argparse
import operator

# In-package imports
import ngram_utils
import stat_utils
import corpus_utils
import sign_utils
import scorers
import bucketers

def parse_profile(profile):
  kargs = {}
  for kv in profile.split(','):
    k, v = kv.split('=')
    kargs[k] = v
  return kargs

def print_score_report(ref, out1, out2,
                       score_type='bleu',
                       bootstrap=0):
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

  if int(bootstrap) > 0:
    print('Significance test. This may take a while.')
    wins, sys1_stats, sys2_stats = sign_utils.eval_with_paired_bootstrap(ref, out1, out2, score_type=score_type, num_samples=int(bootstrap))

    print('Win ratio: Sys1=%.3f, Sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
      print('(Sys1 is superior with p value p=%.3f)\n' % (1-wins[0]))
    elif wins[1] > wins[0]:
      print('(Sys2 is superior with p value p=%.3f)\n' % (1-wins[1]))

    print('Sys1: mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (sys1_stats['mean'], sys1_stats['median'], sys1_stats['lower_bound'], sys1_stats['upper_bound']))
    print('Sys2: mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (sys2_stats['mean'], sys2_stats['median'], sys2_stats['lower_bound'], sys2_stats['upper_bound']))

def print_word_accuracy_report(ref, out1, out2,
                          acc_type='fmeas', bucket_type='freq',
                          freq_count_file=None, freq_corpus_file=None,
                          label_set=None,
                          ref_labels=None, out1_labels=None, out2_labels=None):
  """
  Print a report comparing the word accuracy.

  Args:
    ref: Tokens from the reference
    out1: Tokens from the output file 1
    out2: Tokens from the output file 2
    acc_type: The type of accuracy to show (prec/rec/fmeas). Can also have multiple separated by '+'.
    bucket_type: A string specifying the way to bucket words together to calculate F-measure (freq/tag)
    freq_corpus_file: When using "freq" as a bucketer, which corpus to use to calculate frequency.
                      By default this uses the frequency in the reference test set, but it's often more informative
                      se the frequency in the training set, in which case you specify the path of the target side
                      he training corpus.
    freq_count_file: An alternative to freq_corpus that uses a count file in "word\tfreq" format.
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`.
    out1_labels: output 1 labels. must be specified if ref_labels is specified.
    out2_labels: output 2 labels. must be specified if ref_labels is specified.
  """
  acc_type_map = {'prec': 3, 'rec': 4, 'fmeas': 5}
  bucketer = bucketers.create_word_bucketer_from_profile(bucket_type,
                                                         freq_count_file=freq_count_file,
                                                         freq_corpus_file=freq_corpus_file,
                                                         freq_data=ref,
                                                         label_set=label_set)
  ref_labels = corpus_utils.load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
  out1_labels = corpus_utils.load_tokens(out1_labels) if type(out1_labels) == str else out1_labels
  out2_labels = corpus_utils.load_tokens(out2_labels) if type(out2_labels) == str else out2_labels
  matches1 = bucketer.calc_bucketed_matches(ref, out1, ref_labels=ref_labels, out_labels=out1_labels)
  matches2 = bucketer.calc_bucketed_matches(ref, out2, ref_labels=ref_labels, out_labels=out2_labels)
  acc_types = acc_type.split('+')
  for at in acc_types:
    if at not in acc_type_map:
      raise ValueError(f'Unknown accuracy type {at}')
    aid = acc_type_map[at]
    print(f'--- word {acc_type} by {bucketer.name()} bucket')
    for bucket_str, match1, match2 in zip(bucketer.bucket_strs, matches1, matches2):
      print("{}\t{:.4f}\t{:.4f}".format(bucket_str, match1[aid], match2[aid]))
    print()

def print_sentence_bucketed_report(ref, out1, out2,
                                   bucket_type='score', statistic_type='count',
                                   score_measure='bleu'):
  """
  Print a report of sentences by bucket

  Args:
    ref: Tokens from the reference
    out1: Tokens from the output file 1
    out2: Tokens from the output file 2
    bucket_type: The type of bucketing method to use
    score_measure: If using 'score' as either bucket_type or statistic_type, which scorer to use
  """
  bucketer = bucketers.create_sentence_bucketer_from_profile(bucket_type, score_type=score_measure)
  bc1 = bucketer.create_bucketed_corpus(out1, ref=ref)
  bc2 = bucketer.create_bucketed_corpus(out2, ref=ref)

  if statistic_type == 'count':
    aggregator = lambda out,ref: len(out)
  elif statistic_type == 'score':
    scorer = scorers.create_scorer_from_profile(score_measure)
    aggregator = lambda out,ref: scorer.score_corpus(ref,out)[0]
  else:
    raise ValueError(f'Illegal statistic_type {statistic_type}')

  stats1 = [aggregator(out,ref) for (out,ref) in bc1]
  stats2 = [aggregator(out,ref) for (out,ref) in bc2]

  print(f'--- bucket_type={bucket_type}, statistic_type={statistic_type}, score_measure={score_measure}')
  for bs, s1, s2 in zip(bucketer.bucket_strs, stats1, stats2):
    print(f'{bs}\t{s1}\t{s2}')
  print()

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
    alpha: when sorting n-grams for salient features, the smoothing coefficient. A higher smoothing coefficient
           will result in more frequent phenomena (sometimes this is good).
    compare_type: what type of statistic to compare
                  (match: n-grams that match the reference, over: over-produced ngrams, under: under-produced ngrams)
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`.
                If specified, will aggregate statistics over labels instead of n-grams.
    out1_labels: output 1 labels. must be specified if ref_labels is specified.
    out2_labels: output 2 labels. must be specified if ref_labels is specified.
  """
  print(f'--- min_ngram_length={min_ngram_length}, max_ngram_length={max_ngram_length}')
  print(f'    report_length={report_length}, alpha={alpha}, compare_type={compare_type}')
  if type(ref_labels) == str:
    print(f'    ref_labels={ref_labels}, out1_labels={out1_labels}, out2_labels={out2_labels}')
  print()

  ref_labels = corpus_utils.load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
  out1_labels = corpus_utils.load_tokens(out1_labels) if type(out1_labels) == str else out1_labels
  out2_labels = corpus_utils.load_tokens(out2_labels) if type(out2_labels) == str else out2_labels
  total1, match1, over1, under1 = ngram_utils.compare_ngrams(ref, out1, ref_labels=ref_labels, out_labels=out1_labels,
                                                             min_length=min_ngram_length, max_length=max_ngram_length)
  total2, match2, over2, under2 = ngram_utils.compare_ngrams(ref, out2, ref_labels=ref_labels, out_labels=out2_labels,
                                                             min_length=min_ngram_length, max_length=max_ngram_length)
  if compare_type == 'match':
    scores = stat_utils.extract_salient_features(match1, match2, alpha=alpha)
  elif compare_type == 'over':
    scores = stat_utils.extract_salient_features(over1, over2, alpha=alpha)
  elif compare_type == 'under':
    scores = stat_utils.extract_salient_features(under1, under2, alpha=alpha)
  else:
    raise ValueError(f'Illegal compare_type "{compare_type}"')
  scorelist = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

  print(f'--- {report_length} n-grams that System 1 had higher {compare_type}')
  for k, v in scorelist[:report_length]:
    print('{}\t{} (sys1={}, sys2={})'.format(' '.join(k), v, match1[k], match2[k]))
  print(f'\n--- {report_length} n-grams that System 2 had higher {compare_type}')
  for k, v in reversed(scorelist[-report_length:]):
    print('{}\t{} (sys1={}, sys2={})'.format(' '.join(k), v, match1[k], match2[k]))
  print()

def print_sentence_examples(ref, out1, out2,
                            score_type='sentbleu',
                            report_length=10):
  """
  Print examples of sentences that satisfy some criterion, usually score of one system better

  Args:
    ref: Tokens from the reference
    out1: Tokens from the output file 1
    out2: Tokens from the output file 2
    score_type: The type of scorer to use
    report_length: Number of sentences to print for each system being better or worse
  """
  scorer = scorers.create_scorer_from_profile(score_type)
  sname = scorer.name()
  scorediff_list = []
  for i, (o1, o2, r) in enumerate(zip(out1, out2, ref)):
    s1, str1 = scorer.score_sentence(r, o1)
    s2, str2 = scorer.score_sentence(r, o2)
    scorediff_list.append((s2-s1, s1, s2, str1, str2, i))
  scorediff_list.sort()
  print(f'--- {report_length} sentences where Sys1>Sys2 at {sname}')
  for bdiff, s1, s2, str1, str2, i in scorediff_list[:report_length]:
    print ('sys2-sys1={}, sys1={}, sys2={}\nRef:  {}\nSys1: {}\nSys2: {}\n'.format(bdiff, s1, s2, ' '.join(ref[i]), ' '.join(out1[i]), ' '.join(out2[i])))
  print(f'--- {report_length} sentences where Sys2>Sys1 at {sname}')
  for bdiff, s1, s2, str1, str2, i in scorediff_list[-report_length:]:
    print ('sys2-sys1={}, sys1={}, sys2={}\nRef:  {}\nSys1: {}\nSys2: {}\n'.format(bdiff, s1, s2, ' '.join(ref[i]), ' '.join(out1[i]), ' '.join(out2[i])))

def print_header(header):
  print(f'********************** {header} ************************')

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
  parser.add_argument('--compare_scores', type=str, nargs='*',
                      default=['score_type=bleu,bootstrap=0', 'score_type=length,bootstrap=0'],
                      help="""
                      Compare scores. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'print_score_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_word_accuracies', type=str, nargs='*',
                      default=['bucket_type=freq'],
                      help="""
                      Compare word accuracies by buckets. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'print_word_accuracy_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_sentence_buckets', type=str, nargs='*',
                      default=['bucket_type=score,score_measure=sentbleu',
                               'bucket_type=lengthdiff',
                               'bucket_type=length,statistic_type=score,score_measure=bleu'],
                      help="""
                      Compare sentence counts by buckets. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'print_word_accuracy_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_ngrams', type=str, nargs='*',
                      default=['compare_type=match'],
                      help="""
                      Compare ngrams. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'print_ngram_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_sentence_examples', type=str, nargs='*',
                      default=['score_type=sentbleu'],
                      help="""
                      Compare sentences. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'print_sentence_examples' to see which arguments are available.
                      """)
  args = parser.parse_args()

  ref, out1, out2 = [corpus_utils.load_tokens(x) for x in (args.ref_file, args.out1_file, args.out2_file)]

  # Aggregate scores
  if args.compare_scores:
    print_header('Aggregate Scores')
    for profile in args.compare_scores:
      kargs = parse_profile(profile)
      print_score_report(ref, out1, out2, **kargs)
      print()

  # Word accuracy analysis
  if args.compare_word_accuracies:
    print_header('Word Accuracy Analysis')
    for profile in args.compare_word_accuracies:
      kargs = parse_profile(profile)
      print_word_accuracy_report(ref, out1, out2, **kargs)
      print()

  # Sentence count analysis
  if args.compare_sentence_buckets:
    print_header('Sentence Bucket Analysis')
    for profile in args.compare_sentence_buckets:
      kargs = parse_profile(profile)
      print_sentence_bucketed_report(ref, out1, out2, **kargs)
      print()

  # n-gram difference analysis
  if args.compare_ngrams:
    print_header('N-gram Difference Analysis')
    for profile in args.compare_ngrams:
      kargs = parse_profile(profile)
      print_ngram_report(ref, out1, out2, **kargs)
      print()

  # Sentence example analysis
  if args.compare_sentence_examples:
    print_header('Sentence Example Analysis')
    for profile in args.compare_sentence_examples:
      kargs = parse_profile(profile)
      print_sentence_examples(ref, out1, out2, **kargs)
      print()
