# Overall imports
import argparse
import operator
import numpy as np
import numpy.random as npr
import tempfile

# In-package imports
from compare_mt import __version__
from compare_mt import ngram_utils
from compare_mt import stat_utils
from compare_mt import corpus_utils
from compare_mt import sign_utils
from compare_mt import scorers
from compare_mt import bucketers
from compare_mt import reporters
from compare_mt import arg_utils
from compare_mt import formatting
from compare_mt import cache_utils

source_code_url = 'https://github.com/neulab/compare-mt'

def generate_score_report(
  ref, outs,
  src=None,
  score_type='bleu',
  bootstrap=0, prob_thresh=0.05,
  meteor_directory=None, options=None,
  title=None, 
  case_insensitive=False,
  to_cache=False,
  cache_dicts=None
):
  """
  Generate a report comparing overall scores of system(s) in both plain text and graphs.

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    src: Tokens for the source 
    score_type: A string specifying the scoring type (bleu/length)
    bootstrap: Number of samples for significance test (0 to disable)
    prob_thresh: P-value threshold for significance test
    meteor_directory: Path to the directory of the METEOR code
    options: Options when using external program
    compare_directions: A string specifying which systems to compare 
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
    to_cache: Return a list of computed statistics if True
    cache_dicts: A list of dictionaries that store cached statistics for each output
  """
  # check and set parameters
  bootstrap = int(bootstrap)
  prob_thresh = float(prob_thresh)
  if type(case_insensitive) == str:
    case_insensitive = True if case_insensitive == 'True' else False

  # compute statistics
  scorer = scorers.create_scorer_from_profile(score_type, case_insensitive=case_insensitive, meteor_directory=meteor_directory, options=options)

  cache_key_list = ['scores', 'strs', 'sign_stats']
  scores, strs, sign_stats = cache_utils.extract_cache_dicts(cache_dicts, cache_key_list, len(outs))
  if cache_dicts is None:
    scores, strs = zip(*[scorer.score_corpus(ref, out, src=src) for out in outs])
  
  if to_cache:
    cache_dict = cache_utils.return_cache_dict(cache_key_list, [scores, strs, [scorer.cache_stats(ref, outs[0], src=src)] ])
    return cache_dict

  if bootstrap != 0:
    direcs = []
    for i in range(len(scores)):
      for j in range(i+1, len(scores)):
        direcs.append( (i,j) )
    wins, sys_stats = sign_utils.eval_with_paired_bootstrap(ref, outs, src, scorer, direcs, num_samples=bootstrap, cache_stats=sign_stats)
    wins = list(zip(direcs, wins))
  else:
    wins = sys_stats = None

  # generate reports
  reporter = reporters.ScoreReport(scorer=scorer, scores=scores, strs=strs, 
                                   wins=wins, sys_stats=sys_stats, prob_thresh=prob_thresh, 
                                   title=title)
  reporter.generate_report(output_fig_file=f'score-{score_type}-{bootstrap}',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 

def generate_word_accuracy_report(ref, outs,
                          src=None,
                          acc_type='fmeas', bucket_type='freq', bucket_cutoffs=None,
                          freq_count_file=None, freq_corpus_file=None,
                          label_set=None,
                          ref_labels=None, out_labels=None,
                          title=None,
                          case_insensitive=False,
                          output_bucket_details=False,
                          to_cache=False,
                          cache_dicts=None):
  """
  Generate a report comparing the word accuracy in both plain text and graphs.

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    src: Tokens from the source
    acc_type: The type of accuracy to show (prec/rec/fmeas). Can also have multiple separated by '+'.
    bucket_type: A string specifying the way to bucket words together to calculate F-measure (freq/tag)
    bucket_cutoffs: The boundaries between buckets, specified as a colon-separated string.
    freq_corpus_file: When using "freq" as a bucketer, which corpus to use to calculate frequency.
                      By default this uses the frequency in the reference test set, but it's often more informative
                      to use the frequency in the training set, in which case you specify the path of the
                      training corpus.
    freq_count_file: An alternative to freq_corpus that uses a count file in "word\tfreq" format.
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`.
    out_labels: output labels. must be specified if ref_labels is specified.
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
    output_bucket_details: A boolean specifying whether to output the number of words in each bucket
    to_cache: Return a list of computed statistics if True
    cache_dicts: A list of dictionaries that store cached statistics for each output
  """
  # check and set parameters
  if type(case_insensitive) == str:
    case_insensitive = True if case_insensitive == 'True' else False
  if type(output_bucket_details) == str:
    output_bucket_details = True if output_bucket_details == 'True' else False

  if type(ref_labels) == str:
    ref_labels = corpus_utils.load_tokens(ref_labels)
  if out_labels is not None:
    out_label_files = arg_utils.parse_files(out_labels)
    out_labels = [corpus_utils.load_tokens(x) for x in out_label_files]
    if len(out_labels) != len(outs):
      raise ValueError(f'The number of output files should be equal to the number of output labels.')
    for i, (o, ol) in enumerate(zip(outs, out_labels)):
      if len(o) != len(ol):
        raise ValueError(f'The labels in {out_label_files[i]} do not match the length of the output file {outs[i]}.')

  # compute statistics
  bucketer = bucketers.create_word_bucketer_from_profile(bucket_type,
                                                         bucket_cutoffs=bucket_cutoffs,
                                                         freq_count_file=freq_count_file,
                                                         freq_corpus_file=freq_corpus_file,
                                                         freq_data=ref,
                                                         label_set=label_set,
                                                         case_insensitive=case_insensitive)

  cache_key_list = ['statistics', 'my_ref_total_list', 'my_out_totals_list', 'my_out_matches_list']
  statistics, my_ref_total_list, my_out_totals_list, my_out_matches_list = cache_utils.extract_cache_dicts(cache_dicts, cache_key_list, len(outs))
  if cache_dicts is None:
    statistics, my_ref_total_list, my_out_totals_list, my_out_matches_list = bucketer.calc_statistics(ref, outs, ref_labels=ref_labels, out_labels=out_labels)
  else:
    my_ref_total_list = my_ref_total_list[0]
    my_out_totals_list = list(np.concatenate(my_out_totals_list, 1))
    my_out_matches_list = list(np.concatenate(my_out_matches_list, 1))
  examples = bucketer.calc_examples(len(ref), len(outs), statistics, my_ref_total_list, my_out_matches_list)

  bucket_cnts, bucket_intervals = bucketer.calc_bucket_details(my_ref_total_list, my_out_totals_list, my_out_matches_list) if output_bucket_details else (None, None)

  if to_cache:
    cache_dict = cache_utils.return_cache_dict(cache_key_list, [statistics, [my_ref_total_list], [my_out_totals_list] ,[my_out_matches_list]])
    return cache_dict

  # generate reports
  reporter = reporters.WordReport(bucketer=bucketer,
                                  statistics=statistics,
                                  examples=examples,
                                  bucket_cnts=bucket_cnts,
                                  bucket_intervals=bucket_intervals,
                                  src_sents=src,
                                  ref_sents=ref,
                                  ref_labels=ref_labels,
                                  out_sents=outs,
                                  out_labels=out_labels,
                                  acc_type=acc_type, header="Word Accuracy Analysis",
                                  title=title)
  reporter.generate_report(output_fig_file=f'word-acc',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 
  

def generate_src_word_accuracy_report(ref, outs, src, ref_align_file=None,
                          acc_type='rec', bucket_type='freq', bucket_cutoffs=None,
                          freq_count_file=None, freq_corpus_file=None,
                          label_set=None,
                          src_labels=None,
                          title=None,
                          case_insensitive=False,
                          output_bucket_details=False,
                          to_cache=False,
                          cache_dicts=None):
  """
  Generate a report for source word analysis in both plain text and graphs.

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    src: Tokens from the source
    ref_align_file: Alignment file for the reference
    acc_type: The type of accuracy to show (prec/rec/fmeas). Can also have multiple separated by '+'.
    bucket_type: A string specifying the way to bucket words together to calculate F-measure (freq/tag)
    bucket_cutoffs: The boundaries between buckets, specified as a colon-separated string.
    freq_corpus_file: When using "freq" as a bucketer, which corpus to use to calculate frequency.
                      By default this uses the frequency in the reference test set, but it's often more informative
                      se the frequency in the training set, in which case you specify the path of the target side
                      he training corpus.
    freq_count_file: An alternative to freq_corpus that uses a count file in "word\tfreq" format.
    src_labels: either a filename of a file full of source labels, or a list of strings corresponding to `ref`.
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
    output_bucket_details: A boolean specifying whether to output the number of words in each bucket
    to_cache: Return a list of computed statistics if True
    cache_dicts: A list of dictionaries that store cached statistics for each output
  """
  # check and set parameters
  if type(case_insensitive) == str:
    case_insensitive = True if case_insensitive == 'True' else False
  if type(output_bucket_details) == str:
    output_bucket_details = True if output_bucket_details == 'True' else False

  if acc_type != 'rec':
    raise ValueError("Source word analysis can only use recall as an accuracy type")
  if not src or not ref_align_file:
    raise ValueError("Must specify the source and the alignment file when performing source analysis.")
  if type(src_labels) == str:
    src_labels = corpus_utils.load_tokens(src_labels)

  ref_align = corpus_utils.load_alignments(ref_align_file) 

  # compute statistics
  bucketer = bucketers.create_word_bucketer_from_profile(bucket_type,
                                                         bucket_cutoffs=bucket_cutoffs,
                                                         freq_count_file=freq_count_file,
                                                         freq_corpus_file=freq_corpus_file,
                                                         freq_data=src,
                                                         label_set=label_set,
                                                         case_insensitive=case_insensitive)

  cache_key_list = ['statistics', 'my_ref_total_list', 'my_out_totals_list', 'my_out_matches_list']
  statistics, my_ref_total_list, my_out_totals_list, my_out_matches_list = cache_utils.extract_cache_dicts(cache_dicts, cache_key_list, len(outs))
  if cache_dicts is not None:
    my_ref_total_list = my_ref_total_list[0]
    my_out_totals_list = list(np.concatenate(my_out_totals_list, 1))
    my_out_matches_list = list(np.concatenate(my_out_matches_list, 1))
  else:
    statistics, my_ref_total_list, my_out_totals_list, my_out_matches_list = bucketer.calc_statistics(ref, outs, src=src, src_labels=src_labels, ref_aligns=ref_align)
  examples = bucketer.calc_examples(len(ref), len(outs), statistics, my_ref_total_list, my_out_matches_list)

  bucket_cnts, bucket_intervals = bucketer.calc_bucket_details(my_ref_total_list, my_out_totals_list, my_out_matches_list) if output_bucket_details else (None, None)

  if to_cache:
    cache_dict = cache_utils.return_cache_dict(cache_key_list, [statistics, [my_ref_total_list], [my_out_totals_list], [my_out_matches_list]])
    return cache_dict

  # generate reports
  reporter = reporters.WordReport(bucketer=bucketer,
                                  statistics=statistics,
                                  examples=examples,
                                  bucket_cnts=bucket_cnts,
                                  bucket_intervals=bucket_intervals,
                                  src_sents=src,
                                  ref_sents=ref,
                                  ref_aligns=ref_align,
                                  out_sents=outs,
                                  src_labels=src_labels,
                                  acc_type=acc_type, header="Source Word Accuracy Analysis",
                                  title=title)

  reporter.generate_report(output_fig_file=f'src-word-acc',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 

def generate_sentence_bucketed_report(ref, outs, src=None,
                                   bucket_type='score', bucket_cutoffs=None,
                                   statistic_type='count',
                                   score_measure='sentbleu',
                                   label_set=None,
                                   ref_labels=None, out_labels=None,
                                   title=None,
                                   case_insensitive=False,
                                   output_bucket_details=False,
                                   to_cache=False,
                                   cache_dicts=None):
  """
  Generate a report of sentences by bucket in both plain text and graphs

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    bucket_type: The type of bucketing method to use
    score_measure: If using 'score' as either bucket_type or statistic_type, which scorer to use
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`. Would overwrite out_labels if specified.
    out_labels: output labels. 
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
    output_bucket_details: A boolean specifying whether to output the number of words in each bucket
    to_cache: Return a list of computed statistics if True
    cache_dicts: A list of dictionaries that store cached statistics for each output
  """
  # check and set parameters
  if type(case_insensitive) == str:
    case_insensitive = True if case_insensitive == 'True' else False
  if type(output_bucket_details) == str:
    output_bucket_details = True if output_bucket_details == 'True' else False

  if ref_labels is not None:
    ref_labels = corpus_utils.load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
    if len(ref_labels) != len(ref):
      raise ValueError(f'The number of labels should be equal to the number of sentences.')

  elif out_labels is not None:
    out_labels = arg_utils.parse_files(out_labels)
    if len(out_labels) != len(outs):
      raise ValueError(f'The number of output files should be equal to the number of output labels.')

    out_labels = [corpus_utils.load_tokens(out_label) if type(out_label) == str else out_label for out_label in out_labels]
    for out, out_label in zip(outs, out_labels):
      if len(out_label) != len(out):
        raise ValueError(f'The number of labels should be equal to the number of sentences.')

  # compute statistics
  bucketer = bucketers.create_sentence_bucketer_from_profile(bucket_type, bucket_cutoffs=bucket_cutoffs,
                                                             score_type=score_measure, label_set=label_set, case_insensitive=case_insensitive)

  src = [None for _ in ref] if src is None else src

  if statistic_type == 'count':
    scorer = None
    if bucket_type != 'score' and bucket_type != 'lengthdiff':
      ref = ref_label = None
    aggregator = lambda out,refs,src: len(out)
  elif statistic_type == 'score':
    scorer = scorers.create_scorer_from_profile(score_measure, case_insensitive=case_insensitive)
    aggregator = lambda out,ref,src: scorer.score_corpus(ref,out,src)[0]
  else:
    raise ValueError(f'Illegal statistic_type {statistic_type}')
  

  cache_key_list = ['stats']
  stats = cache_utils.extract_cache_dicts(cache_dicts, cache_key_list, len(outs))

  if cache_dicts is None:
    bcs = [bucketer.create_bucketed_corpus(out, ref=ref, src=src, ref_labels=ref_labels if ref_labels else None, out_labels=out_labels[i] if out_labels else None) for i, out in enumerate(outs)]
    stats = [[aggregator(out,ref,src) for (out,ref,src) in bc] for bc in bcs]

  if output_bucket_details and statistic_type == 'score':
    bucket_cnt_calculator = lambda out,ref,src: len(out)
    bucket_interval_calculator = lambda out,ref: sign_utils.eval_with_paired_bootstrap(ref, [out], src, scorer, None)[1][0]
    if cache_dicts is not None: # we don't cache bcs
      bcs = [bucketer.create_bucketed_corpus(out, ref=ref, src=src,ref_labels=ref_labels if ref_labels else None, out_labels=out_labels[i] if out_labels else None) for i, out in enumerate(outs)]
    bucket_cnts = [bucket_cnt_calculator(out,ref,src) for (out,ref,src) in bcs[0]]
    bucket_intervals = [[bucket_interval_calculator(out,ref,src) for (out,ref,src) in bc] for bc in bcs]
  else:
    bucket_cnts = bucket_intervals = None
  

  if to_cache:
    cache_dict = cache_utils.return_cache_dict(cache_key_list, [stats])
    return cache_dict

  # generate reports
  reporter = reporters.SentenceReport(bucketer=bucketer,
                                      sys_stats=stats,
                                      statistic_type=statistic_type, scorer=scorer, 
                                      bucket_cnts=bucket_cnts,
                                      bucket_intervals=bucket_intervals,
                                      title=title)

  reporter.generate_report(output_fig_file=f'sentence-{statistic_type}-{score_measure}',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 
  

def generate_ngram_report(ref, outs,
                       min_ngram_length=1, max_ngram_length=4,
                       report_length=50, alpha=1.0, compare_type='match',
                       ref_labels=None, out_labels=None,
                       compare_directions='0-1',
                       title=None,
                       case_insensitive=False,
                       to_cache=False,
                       cache_dicts=None):
  """
  Generate a report comparing aggregate n-gram statistics in both plain text and graphs

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    min_ngram_length: minimum n-gram length
    max_ngram_length: maximum n-gram length
    report_length: the number of n-grams to report
    alpha: when sorting n-grams for salient features, the smoothing coefficient. A higher smoothing coefficient
           will result in more frequent phenomena (sometimes this is good).
    compare_type: what type of statistic to compare
                  (match: n-grams that match the reference, over: over-produced ngrams, under: under-produced ngrams)
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`.
                If specified, will aggregate statistics over labels instead of n-grams.
    out_labels: output labels. must be specified if ref_labels is specified.
    compare_directions: A string specifying which systems to compare
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
    to_cache: Return a list of computed statistics if True
    cache_dicts: A list of dictionaries that store cached statistics for each output
  """
  # check and set parameters
  min_ngram_length, max_ngram_length, report_length = int(min_ngram_length), int(max_ngram_length), int(report_length)
  alpha = float(alpha) if type(alpha) == str else alpha
  if type(case_insensitive) == str:
    case_insensitive = True if case_insensitive == 'True' else False

  if out_labels is not None:
    out_labels = arg_utils.parse_files(out_labels)
    if len(out_labels) != len(outs):
      raise ValueError(f'The number of output files should be equal to the number of output labels.')

  if type(ref_labels) == str:
    label_files_str = f'    ref_labels={ref_labels},'
    for i, out_label in enumerate(out_labels):
      label_files_str += f' out{i}_labels={out_label},'
    label_files = (label_files_str)
  else:
    label_files = None

  # compute statistics
  cache_key_list = ['totals', 'matches', 'overs', 'unders']
  totals, matches, overs, unders = cache_utils.extract_cache_dicts(cache_dicts, cache_key_list, len(outs))
  if cache_dicts is None:
    if not type(ref_labels) == str and case_insensitive:
      ref = corpus_utils.lower(ref)
      outs = [corpus_utils.lower(out) for out in outs]

    ref_labels = corpus_utils.load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
    out_labels = [corpus_utils.load_tokens(out_labels[i]) if not out_labels is None else None for i in range(len(outs))]
    totals, matches, overs, unders = zip(*[ngram_utils.compare_ngrams(ref, out, ref_labels=ref_labels, out_labels=out_label,
                                                             min_length=min_ngram_length, max_length=max_ngram_length) for out, out_label in zip(outs, out_labels)])

  if to_cache:
    cache_dict = cache_utils.return_cache_dict(cache_key_list, [totals, matches, overs, unders])
    return cache_dict

  direcs = arg_utils.parse_compare_directions(compare_directions)
  scores = []
  for (left, right) in direcs:
    if compare_type == 'match':
      scores.append(stat_utils.extract_salient_features(matches[left], matches[right], alpha=alpha))
    elif compare_type == 'over':
      scores.append(stat_utils.extract_salient_features(overs[left], overs[right], alpha=alpha))
    elif compare_type == 'under':
      scores.append(stat_utils.extract_salient_features(unders[left], unders[right], alpha=alpha))
    else:
      raise ValueError(f'Illegal compare_type "{compare_type}"')
  scorelist = [sorted(score.items(), key=operator.itemgetter(1), reverse=True) for score in scores]

  # generate reports
  reporter = reporters.NgramReport(scorelist=scorelist, report_length=report_length,
                                   min_ngram_length=min_ngram_length,
                                   max_ngram_length=max_ngram_length,
                                   matches=matches,
                                   compare_type=compare_type, alpha=alpha,
                                   compare_directions=direcs,
                                   label_files=label_files,
                                   title=title)
  reporter.generate_report(output_fig_file=f'ngram-min{min_ngram_length}-max{max_ngram_length}-{compare_type}',
                           output_fig_format='pdf',
                           output_directory='outputs')
  return reporter

def generate_sentence_examples(ref, outs, src=None,
                            score_type='sentbleu',
                            report_length=10,
                            compare_directions='0-1',
                            title=None,
                            case_insensitive=False,
                            to_cache=False,
                            cache_dicts=None):
  """
  Generate examples of sentences that satisfy some criterion, usually score of one system better

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    src: Tokens from the source (optional)
    score_type: The type of scorer to use
    report_length: Number of sentences to print for each system being better or worse
    compare_directions: A string specifying which systems to compare
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
    to_cache: Return a list of computed statistics if True
    cache_dicts: A list of dictionaries that store cached statistics for each output
  """
  # check and set parameters
  report_length = int(report_length)
  if type(case_insensitive) == str:
    case_insensitive = True if case_insensitive == 'True' else False

    
  # compute statistics
  scorer = scorers.create_scorer_from_profile(score_type, case_insensitive=case_insensitive)

  cache_key_list = ['scores', 'strs']
  scores, strs = cache_utils.extract_cache_dicts(cache_dicts, cache_key_list, len(outs))
  src = [None for _ in ref] if src is None else src
  if cache_dicts is None:
    scores, strs = [], []
    for out in outs:
      scores_i, strs_i = [], []
      for (r, o, s) in zip(ref, out, src):
        score, string = scorer.score_sentence(r, o, s)
        scores_i.append(score)
        strs_i.append(string)
      scores.append(scores_i)
      strs.append(strs_i)
  
  if to_cache:
    cache_dict = cache_utils.return_cache_dict(cache_key_list, [scores, strs])
    return cache_dict

  direcs = arg_utils.parse_compare_directions(compare_directions)

  scorediff_lists = []
  for (left, right) in direcs:
    scorediff_list = []
    deduplicate_set = set()
    for i, (o1, o2, r) in enumerate(zip(outs[left], outs[right], ref)):
      if (tuple(o1), tuple(o2), tuple(r)) in deduplicate_set:
        continue
      deduplicate_set.add( (tuple(o1), tuple(o2), tuple(r)) )
      s1, str1 = scores[left][i], strs[left][i]
      s2, str2 = scores[right][i], strs[right][i]
      scorediff_list.append((s2-s1, s1, s2, str1, str2, i))
    scorediff_list.sort()
    scorediff_lists.append(scorediff_list)

  # generate reports
  reporter = reporters.SentenceExampleReport(report_length=report_length, scorediff_lists=scorediff_lists,
                                             scorer=scorer,
                                             ref=ref, outs=outs, src=src,
                                             compare_directions=direcs,
                                             title=title)
  reporter.generate_report()
  return reporter 

def main():
  parser = argparse.ArgumentParser(
      description='Program to compare MT results',
      epilog=f'For more details, see {source_code_url}'
  )
  parser.add_argument('ref_file', type=str,
                      help='A path to a correct reference file')
  parser.add_argument('out_files', type=str, nargs='+',
                      help='Paths to system outputs')
  parser.add_argument('--sys_names', type=str, nargs='+', default=None,
                      help='Names for each system, must be same number as output files')
  parser.add_argument('--src_file', type=str, default=None,
                      help='A path to the source file')
  parser.add_argument('--fig_size', type=str, default='6x4.5',
                      help='The size of figures, in "width x height" format.')
  parser.add_argument('--compare_scores', type=str, nargs='*',
                      default=['score_type=bleu', 'score_type=length'],
                      help="""
                      Compare scores. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_score_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_word_accuracies', type=str, nargs='*',
                      default=['bucket_type=freq'],
                      help="""
                      Compare word accuracies by buckets. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_word_accuracy_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_src_word_accuracies', type=str, nargs='*',
                      default=None,
                      help="""
                      Source analysis. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_src_word_accuracy_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_sentence_buckets', type=str, nargs='*',
                      default=['bucket_type=length,statistic_type=score,score_measure=bleu',
                               'bucket_type=lengthdiff',
                               'bucket_type=score,score_measure=sentbleu'],
                      help="""
                      Compare sentence counts by buckets. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_sentence_buckets_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_ngrams', type=str, nargs='*',
                      default=['compare_type=match'],
                      help="""
                      Compare ngrams. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_ngram_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_sentence_examples', type=str, nargs='*',
                      default=['score_type=sentbleu'],
                      help="""
                      Compare sentences. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_sentence_examples' to see which arguments are available.
                      """)
  parser.add_argument('--output_directory', type=str, default=None,
                      help="""
                      A path to a directory where a graphical report will be saved. Open index.html in the directory
                      to read the report.
                      """)
  parser.add_argument('--report_title', type=str, default='compare-mt Analysis Report',
                      help="""
                      The name of the HTML report.
                      """)
  parser.add_argument('--decimals', type=int, default=4,
                      help="Number of decimals to print for floating point numbers")
  parser.add_argument('--seed', type=int, default=None,
                      help="Seed for random number generation")
  parser.add_argument('--scorer_scale', type=float, default=100, choices=[1, 100],
                      help="Set the scale of BLEU, METEOR, WER, chrF and COMET to 0-1 or 0-100 (default 0-100)")
  parser.add_argument('--http', type=int, dest='bind_port',
                      help='Launch an HTTP server at specified port to view results.'
                           'Disabled by default, but specifying a port number enabled it.')
  parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
  args = parser.parse_args()

  # Set formatting
  formatting.fmt.set_decimals(args.decimals)

  # Set random seed
  if args.seed is not None:
    npr.seed(args.seed)

  # Set scale
  scorers.global_scorer_scale = args.scorer_scale

  ref = corpus_utils.load_tokens(args.ref_file)
  outs = [corpus_utils.load_tokens(x) for x in args.out_files]

  src = corpus_utils.load_tokens(args.src_file) if args.src_file else None 
  reporters.sys_names = args.sys_names if args.sys_names else [f'sys{i+1}' for i in range(len(outs))]
  reporters.fig_size = tuple([float(x) for x in args.fig_size.split('x')])
  if len(reporters.sys_names) != len(outs):
    raise ValueError(f'len(sys_names) != len(outs) -- {len(reporters.sys_names)} != {len(outs)}')

  reports = []

  report_types = [
    (args.compare_scores, generate_score_report, 'Aggregate Scores', True),
    (args.compare_word_accuracies, generate_word_accuracy_report, 'Word Accuracies', False),
    (args.compare_src_word_accuracies, generate_src_word_accuracy_report, 'Source Word Accuracies', True),
    (args.compare_sentence_buckets, generate_sentence_bucketed_report, 'Sentence Buckets', True)]
  if len(outs) > 1:
    report_types += [
      (args.compare_ngrams, generate_ngram_report, 'Characteristic N-grams', False),
      (args.compare_sentence_examples, generate_sentence_examples, 'Sentence Examples', True),
    ]

  for arg, func, name, use_src in report_types:
    if arg is not None:
      if use_src:
        reports.append( (name, [func(ref, outs, src, **arg_utils.parse_profile(x)) for x in arg]) )
      else:
        reports.append( (name, [func(ref, outs, **arg_utils.parse_profile(x)) for x in arg]) )

  # Write all reports into a single html file
  if args.output_directory != None:
    reporters.generate_html_report(reports, args.output_directory, args.report_title)

  if args.bind_port:
    out_dir = args.output_directory
    if not out_dir:
      out_dir = tempfile.TemporaryDirectory().name
      reporters.generate_html_report(reports, out_dir, args.report_title)
    reporters.launch_http_server(out_dir, bind_port=args.bind_port)


if __name__ == '__main__':
  main()
