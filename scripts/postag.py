# This is a simple script to POS tag already-tokenized English using NLTK. To run it just do:
# $ python postag.py < file.eng > file.eng.tag
# You may need to install the NLTK POS tagger if you haven't already, in which case you'll get an error telling you how
# to do so the first time you run this script.

import nltk
import sys

for line in sys.stdin:
  text = line.strip('\n').split(' ')
  print(' '.join([x[1] for x in nltk.pos_tag(text)]))
