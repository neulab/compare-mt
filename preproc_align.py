import argparse
from collections import defaultdict, Counter

"""
Heuristic source target lexicon induction by fast_align

options:
  comb_corpora: combine source and target training file for fast_align
    args:
      src_file: path to source training data
      trg_file: path to target training data
      out_file: combined src-trg file 
  get_lex: get src-trg lexicon based on fast_align result
    args:
      src_file: path to source training data
      trg_file: path to target training data
      align_file: path to the output of fast_align on src and trg training data
      out_file: path to the output lexicon of src-trg
  joint_lex: given two lexicons with same target side src1-trg and src2-trg, get src1-src2 
    args:
      lex1_file: path to src1-trg lexicon of language 1
      lex2_file: path to src2-trg lexicon of language 2
      out_file: path to the src1-src2 lexicon

"""
parser = argparse.ArgumentParser()

parser.add_argument("--src_file", type=str)
parser.add_argument("--trg_file", type=str)
parser.add_argument("--out_file", type=str)
parser.add_argument("--align_file", type=str)
#parser.add_argument("--lan_code", type=str)
parser.add_argument("--option", default='comb_corpora', type=str, help="[comb_corpora|get_lex|joint_lex]")

parser.add_argument("--lex1_file", type=str)
parser.add_argument("--lex2_file", type=str)

#parser.add_argument("--l1", type=str)
#parser.add_argument("--l2", type=str)
args = parser.parse_args()

def comb_corpora(src_file, trg_file, out_file):
  src = open(src_file, 'r', encoding='utf-8')
  trg = open(trg_file, 'r', encoding='utf-8')
  out = open(out_file, 'w', encoding='utf-8')
  for s, t in zip(src, trg):
    out.write("{} ||| {}\n".format(s.strip(), t.strip()))

def get_dict_align(align_file, src_file, trg_file, out_file):
  align = open(align_file, 'r', encoding='utf-8')
  src = open(src_file, 'r', encoding='utf-8')
  trg = open(trg_file, 'r', encoding='utf-8')
  out = open(out_file, 'w', encoding='utf-8')
  trg2src = defaultdict(list)
  for a, s, t in zip(align, src, trg):
    s = s.split()
    t = t.split()
    for m in a.split():
      m = m.split('-')
      idxs, it = int(m[0]), int(m[1])
      ws, wt = s[idxs], t[it]
      trg2src[wt].append(ws)
  for tw, sw_list in trg2src.items():
    sw_count = Counter(sw_list)
    sw, sw_c= sw_count.most_common(1)[0]
    if sw_c < 2: continue
    out.write("{} ||| {}\n".format(sw, tw))

def get_joint_lex(lex1, lex2, out):
  lex1 = open(lex1, 'r', encoding='utf-8')
  lex2 = open(lex2, 'r', encoding='utf-8')
  out = open(out, 'w', encoding='utf-8')
  t2s_l1 = {}
  #t2s_l2 = {}
  for line in lex1:
    toks = line.split(' ||| ')
    t2s_l1[toks[1].strip()] = toks[0].strip()
  for line in lex2:
    toks = line.split(' ||| ')
    #t2s_l2[toks[1].strip()] = toks[0].strip()
    t, s = toks[1].strip(), toks[0].strip()
    if t in t2s_l1:
      out.write("{} ||| {}\n".format(t2s_l1[t], s))

if __name__ == "__main__":
  src = args.src_file
  trg = args.trg_file
  out = args.out_file
  #if args.lan_code:
  #  src = "../data/{}_eng/ted-train.mtok.{}".format(args.lan_code, args.lan_code)
  #  trg = "../data/{}_eng/ted-train.mtok.eng".format(args.lan_code)
  if args.option == "comb_corpora":
    #if args.lan_code:
    #  out = "../data/{}_eng/ted-train.mtok.{}-eng".format(args.lan_code, args.lan_code)
    comb_corpora(src, trg, out)
  elif args.option == "get_lex":
    align = args.align_file
    #if args.lan_code:
    #  align = "{}.forward.align".format(args.lan_code)
    #  out = "../data/{}_eng/ted-train.mtok.lex.{}-eng".format(args.lan_code, args.lan_code)
    get_dict_align(align, src, trg, out)
  elif args.option == "joint_lex":
    lex1 = args.lex1_file
    lex2 = args.lex2_file
    #lex1 = "../data/{}_eng/ted-train.mtok.lex.{}-eng".format(args.l1, args.l1)
    #lex2 = "../data/{}_eng/ted-train.mtok.lex.{}-eng".format(args.l2, args.l2)
    #out = "{}{}.lex".format(args.l1, args.l2)
    get_joint_lex(lex1, lex2, out)
