from collections import defaultdict
import math
import itertools
import re
import _pickle as pkl

import numpy as np

"""
This script provides utility function and a few examples
of analyzing NMT output based on features of source words
"""

"""
Analysis functions for calculating the f1 and BLEU score for each bucket 
"""
def f1_by_bucket(w2bucket, num_bucket, ref_lines, hyp_lines):
  matches = [[0,0,0] for x in range(num_bucket+1)]
  ref = [l.split() for l in ref_lines]
  out = [l.split() for l in hyp_lines]
  for refsent, outsent in zip(ref, out):
    reffreq, outfreq = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for x in refsent:
      reffreq[x] += 1
    for x in outsent:
      outfreq[x] += 1
    for k in set(itertools.chain(reffreq.keys(), outfreq.keys())):
      for  match in matches:
        if k in w2bucket:
          bucket = w2bucket[k]
          match = matches[bucket]
          match[0] += min(reffreq[k], outfreq[k])
          match[1] += reffreq[k]
          match[2] += outfreq[k]
          break 
  rec_l, prec_l, f_l = [], [], []
  for bothf, reff, outf in matches:
    if bothf == 0:
      rec, prec, fmeas = 0.0, 0.0, 0.0
    else:
      rec = bothf / float(reff)
      prec = bothf / float(outf)
      fmeas = 2 * prec * rec / (prec + rec)
    #yield bothf, reff, outf, rec, prec, fmeas
    rec_l.append(rec)   
    prec_l.append(prec)   
    f_l.append(fmeas)   
  return rec_l, prec_l, f_l

def bleu_by_bucket(buck_id_dict, hyp_lines, ref_lines):
  buck_id_dict = sorted(buck_id_dict.items(), key=lambda kv: kv[0])
  hyp_bleu = []
  for buck_id, idx_list in buck_id_dict:
    print("buck_id={}, count={}".format(buck_id, len(idx_list)))
    cur_refs = [ref_lines[i] for i in idx_list]
    cur_hyps = [hyp_lines[i] for i in idx_list]
    b = bleu(cur_hyps, [cur_refs], 4)
    hyp_bleu.append(b)
  return hyp_bleu

"""
Example analysis 1: Bucket by number of BPE pieces of source words
"""
def get_sent_piece_num(w2id, bpe_file, tok_file):
  def sent_piece_segs_post(p):
    '''
    Segment a sentence piece string into list of piece string for each word
    '''
    toks = re.compile(r'\u2581')
    ret = []
    p_start = 0
    for m in toks.finditer(p):
      pos = m.start()
      if pos == 0:
        continue
      ret.append(p[p_start:pos + 1].strip())
      p_start = pos + 1
    if p_start != len(p) - 1:
      ret.append(p[p_start:])
    return ret

  bpe_file = open(bpe_file, 'r', encoding='utf-8')
  tok_file = open(tok_file, 'r', encoding='utf-8')
  sp_count = []
  w_count = defaultdict(list)
  for bpe_line, tok_line in zip(bpe_file, tok_file):
    pieces = sent_piece_segs_post(bpe_line)
    words = tok_line.split() 
    count = []
    for p in pieces:
      count.append(len([i for i in p.split() if i != "▁" ]))
    sp_count.append(count)
    for w, c in zip(words, count):
      if w in w2id:
        w_count[w].append(c)
      else:
        print(w)
  return sp_count, w_count

def bpe_num_f1(num_bucket, src_vocab, src_eng_lex, hyp1_file, hyp2_file, src_file, src_bpe, ref_file):
  ref_lines = open(ref_file, 'r', encoding='utf-8').readlines()
  src_lines = open(src_file, 'r', encoding='utf-8').readlines()
  hyp1_lines = open(hyp1_file, 'r', encoding='utf-8').readlines()
  hyp2_lines = open(hyp2_file, 'r', encoding='utf-8').readlines()

  eng2src = {}
  src2eng = {}
  with open(src_eng_lex, 'r', encoding='utf-8') as myfile:
    for line in myfile:
      toks = line.split(" ||| ")
      s, e = toks[0].strip(), toks[1].strip()
      eng2src[e] = s
      src2eng[s] = e

  w2id = {}
  with open(src_vocab, 'r', encoding='utf-8') as myfile:
    for line in myfile:
      w2id[line.strip()] = len(w2id)
  sp_count, w_count = get_sent_piece_num(w2id,src_bpe, src_file)

  w2bucket = {}
  for w, count in w_count.items():
    if len(count) == 0:
      print(w)
    else:
      if w in src2eng:
        tw = src2eng[w]
        piece_n = int(sum(count)/len(count))
        if piece_n == 1:
          w2bucket[tw] = 1
        else:
          w2bucket[tw] = 2
        w2bucket[tw] = piece_n
        #print(w, tw, w2bucket[tw])
  i = max(w2bucket.values())

  print("start calculating f1")
  rec1, pre1, fmea1 = f1_by_bucket(w2bucket, i, ref_lines, hyp1_lines)
  print("finished f1 for sys1")
  rec2, pre2, fmea2 = f1_by_bucket(w2bucket, i, ref_lines, hyp2_lines)
  print("finished f1 for sys2")
  print("recall")
  print("sys1 sys2 sys1-sys2")
  bucket = 0
  for b1, b2 in zip(rec1, rec2):
    print("bucket {}".format(bucket))
    print(b1, b2, b1-b2)
    bucket += 1

  print("precision")
  print("sys1 sys2 sys1-sys2")
  bucket = 0
  for b1, b2 in zip(pre1, pre2):
    print("bucket {}".format(bucket))
    print(b1, b2, b1-b2)
    bucket += 1

  print("f1")
  print("sys1 sys2 sys1-sys2")
  bucket = 0
  for b1, b2 in zip(fmea1, fmea2):
    print("bucket {}".format(bucket))
    print(b1, b2, b1-b2)
    bucket += 1

"""
Example analysis 2: Bucket by edit distance of source words
"""

def edit_dist_f1(num_bucket, dist_lex_file,  src_vocab, src_eng_lex, hyp1_file, hyp2_file, src_file, ref_file):
  ref_lines = open(ref_file, 'r', encoding='utf-8').readlines()
  src_lines = open(src_file, 'r', encoding='utf-8').readlines()
  hyp1_lines = open(hyp1_file, 'r', encoding='utf-8').readlines()
  hyp2_lines = open(hyp2_file, 'r', encoding='utf-8').readlines()
  eng2src = {}
  src2eng = {}
  with open(src_eng_lex, 'r', encoding='utf-8') as myfile:
    for line in myfile:
      toks = line.split(" ||| ")
      s, e = toks[0].strip(), toks[1].strip()
      eng2src[e] = s
      src2eng[s] = e
  w2rank = {}
  with open(dist_lex_file, encoding='utf-8') as myfile:
    for line in myfile:
      toks = line.split(" ||| ")
      sw, rw, dist = toks[0].strip(), toks[1].strip(), float(toks[2].strip())
      if sw in src2eng and dist < 5:
        tw = src2eng[sw]
        w2rank[tw] = dist
  w2rank = sorted(w2rank.items(), key=lambda kv: kv[1])
  w2bucket = {}
  n_per_bucket = len(w2rank) // num_bucket
  i, start = 0, 0
  while start < len(w2rank):
    end = min(start+n_per_bucket, len(w2rank))
    for kv in w2rank[start:end]:
      w2bucket[kv[0]] = i
    start = end
    i += 1
  #print(w2bucket)
  print("start calculating f1")
  rec1, pre1, fmea1 = f1_by_bucket(w2bucket, i, ref_lines, hyp1_lines)
  print("finished f1 for sys1")
  rec2, pre2, fmea2 = f1_by_bucket(w2bucket, i, ref_lines, hyp2_lines)
  print("finished f1 for sys2")
  print("recall")
  print("sys1 sys2 sys1-sys2")
  bucket = 0
  for b1, b2 in zip(rec1, rec2):
    print("bucket {}".format(bucket))
    print(b1, b2, b1-b2)
    bucket += 1

  print("precision")
  print("sys1 sys2 sys1-sys2")
  bucket = 0
  for b1, b2 in zip(pre1, pre2):
    print("bucket {}".format(bucket))
    print(b1, b2, b1-b2)
    bucket += 1

  print("f1")
  print("sys1 sys2 sys1-sys2")
  bucket = 0
  for b1, b2 in zip(fmea1, fmea2):
    print("bucket {}".format(bucket))
    print(b1, b2, b1-b2)
    bucket += 1

if __name__ == "__main__":
  dist_lex = "data/example/src1-src2.lex-dist" 
  src_vocab = "data/example/src-vocab"
  src_eng_lex = "data/example/src-trg.lex"
  hyp1_file = "data/example/sys1-hyp"
  hyp2_file = "data/example/sys2-hyp"
  src_file = "data/example/src"
  src_bpe = "data/example/src-bpe"
  ref_file = "data/example/trg"

  edit_dist_f1(5, dist_lex, src_vocab, src_eng_lex, hyp1_file, hyp2_file, src_file, ref_file)
  #bpe_num_f1(5, src_vocab, src_eng_lex, hyp1_file, hyp2_file, src_file, src_bpe, ref_file)
