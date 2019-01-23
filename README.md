# compare_mt
by Graham Neubig (and other contributors!)

[![Build Status](https://travis-ci.org/neulab/compare-mt.svg?branch=master)](https://travis-ci.org/neulab/compare-mt)

This is a script to compare the output of multiple machine translation or language generation systems.
To use it you to have, in text format, a "correct" reference, and the output of two different systems.
Based on this, `compare_mt.py` will run a number of analyses that attempt to pick out salient differences between
the systems, which will make it easier for you to figure out what things one system is doing better tha another.

## Basic Usage

First, you need to install the requirements:

    pip install -r requirements.txt

Then, as an example, you can run this over two included system outputs.

    python compare_mt.py example/ted.ref.eng example/ted.sys1.eng example/ted.sys2.eng

Here, system 1 and system 2 are the baseline phrase-based and neural Slovak-English systems from our
[EMNLP 2018 paper](http://aclweb.org/anthology/D18-1103). This will print out a number of statistics including:

* **Aggregate Scores:** A report on overall BLEU scores and length ratios
* **Word Accuracy Analysis:** A report on the F-measure of words by frequency bucket
* **Sentence Bucket Analysis:** Bucket sentences by various statistics (e.g. sentence BLEU, length difference with the
  reference, overall length), and calculate statistics by bucket (e.g. number of sentences, BLEU score per bucket)
* **N-gram Difference Analysis:** Calculate which n-grams one system is consistently translating better
* **Sentence Examples:** Find sentences where one system is doing better than the other according to sentence BLEU

Try running it first and taking a look to see what you find. To summarize the results that immediately stick out here:

* From the *aggregate scores* we can see that the BLEU of neural MT is higher, but its sentences are slightly shorter.
* From the *word accuracy analysis* we can see that phrase-based MT is better at low-frequency words.
* From the *sentence bucket analysis* we can see that neural seems to be better at translating shorter sentences.
* From the *n-gram difference analysis* we can see that there are a few words that neural MT is not good at
  but phrase based MT gets right (e.g. "phantom"), while there are a few long phrases that neural MT does better with
  (e.g. "going to show you").

If you run on your own data, you might be able to find more interesting things about your own systems. Try comparing
your modified system with your baseline and seeing what you find! 

## Other Options

There are many options that can be used to do different types of analysis.
If you want to find all the different types of analysis supported, the most comprehensive way to do so is by
taking a look at `compare_mt.py`, which is documented relatively well and should give examples.
We do highlight a few particularly useful and common types of analysis below:

### Significance Tests

The script allows you to perform statistical significance tests for scores based on bootstrap resampling. You can set
the number of samplings manually. Here is an example using the example data:

    python compare_mt.py example/ted.ref.eng example/ted.sys1.eng example/ted.sys2.eng --compare_scores score_type=bleu,bootstrap=1000

### Using Training Set Frequency

One useful piece of analysis is the "word accuracy by frequency" analysis. By default this frequency is the frequency
in the *test set*, but arguably it is more informative to know accuracy by frequency in the *training set* as this
demonstrates the models' robustness to words they haven't seen much, or at all, in the training data. To change the
corpus used to calculate word frequency and use the training set (or some other set), you can set the `freq_corpus_file`
option to the appropriate corpus.

    python compare_mt.py example/ted.ref.eng example/ted.sys1.eng example/ted.sys2.eng
        --compare_word_accuracies bucket_type=freq,freq_corpus_file=example/ted.train.eng

### Incorporating Word Labels

If you're interested in performing aggregate analysis over labels for each word instead of the words themselves, it
is possible to do so. As an example, we've included POS tags for each of the example outputs. You can use these in
aggregate analysis, or n-gram-based analysis. The following gives an example:

    python compare_mt.py example/ted.ref.eng example/ted.sys1.eng example/ted.sys2.eng
        --compare_word_accuracies bucket_type=label,ref_labels=example/ted.ref.eng.tag,out1_labels=example/ted.sys1.eng.tag,out2_labels=example/ted.sys2.eng.tag,label_set=CC+DT+IN+JJ+NN+NNP+NNS+PRP+RB+TO+VB+VBP+VBZ
        --compare_ngrams compare_type=match,ref_labels=example/ted.ref.eng.tag,out1_labels=example/ted.sys1.eng.tag,out2_labels=example/ted.sys2.eng.tag

This will calculate word accuracies and n-gram matches by POS bucket, and allows you to see things like the fact
that the phrase-based MT system is better at translating content words such as nouns and verbs, while neural MT
is doing better at translating function words.

### Analyzing Source Words

If you have a source corpus that is aligned to the target, you can also analyze accuracies according to features of the
source language words, which would allow you to examine whether, for example, infrequent words on the source side are
hard to output properly. Here is an example using the example data:

    python compare_mt.py example/ted.ref.eng example/ted.sys1.eng example/ted.sys2.eng --src_file example/ted.orig.slk
        --compare_src_word_accuracies ref_align=example/ted.ref.align,out1_align=example/ted.sys1.align,out2_align=example/ted.sys2.align

### Analyzing Word Likelihoods

If you wish to analyze the word log likelihoods by two systems on the target corpus, you can use the following

    python compare_ll.py --ref example/ll_test.txt --ll1-file example/ll_test.sys1.likelihood --ll2-file example/ll_test.sys2.likelihood --compare-word-likelihoods bucket_type=freq,freq_corpus_file=example/ll_test.txt

You can analyze the word log likelihoods over labels for each word instead of the words themselves:

    python compare_ll.py --ref example/ll_test.txt --ll1-file example/ll_test.sys1.likelihood --ll2-file example/ll_test.sys2.likelihood --compare-word-likelihoods bucket_type=label,label_corpus=example/ll_test.tag,label_set=CC+DT+IN+JJ+NN+NNP+NNS+PRP+RB+TO+VB+VBP+VBZ

NOTE: You can also use the above to also analyze the word likelihoods produced by two language models.

## Citation/References

There is currently no canonical reference for `compare_mt.py`, but particularly the n-gram difference analysis
is loosely based on

* Akabe et al. "[Discriminative Language Models as a Tool for Machine Translation Error Analysis](http://www.phontron.com/paper/akabe14coling.pdf)" COLING 2014.

If you're interested in adding a citation, we'd appreciate the following one:


    @inproceedings{akabe14coling,
        title = {Discriminative Language Models as a Tool for Machine Translation Error Analysis},
        author = {Koichi Akabe and Graham Neubig and Sakriani Sakti and Tomoki Toda and Satoshi Nakamura},
        booktitle = {The 25th International Conference on Computational Linguistics (COLING)},
        address = {Dublin, Ireland},
        month = {August},
        pages = {1124--1132},
        url = {http://www.phontron.com/paper/akabe14coling.pdf},
        year = {2014}
    }

It also borrows ideas from some of the following papers:

* **Automatic Error Analysis:**
  Popovic and Ney "[Towards Automatic Error Analysis of Machine Translation Output](https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00072)" Computational Linguistics 2011.
* **POS-based Analysis:**
  Chiang et al. "[The Hiero Machine Translation System](http://aclweb.org/anthology/H05-1098)" EMNLP 2005.

There is also other good software for automatic comparison or error analysis of MT systems:

* **[MT-ComparEval](https://github.com/choko/MT-ComparEval):** Very nice for visualization of individual examples, but
  not as focused on aggregate analysis as `compare_mt.py`. Also has more software dependencies and requires using a web
  browser, while `compare_mt.py` can be used as a command-line tool.
