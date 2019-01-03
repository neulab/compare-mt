# compare-mt
by Graham Neubig

This is a script to compare MT results, loosely based [Discriminative Language Models as a Tool for Machine Translation Error Analysis](http://www.phontron.com/paper/akabe14coling.pdf) (Akabe et al. 2014):

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

## Usage Instructions

Run this over two system outputs, as follows:

    python compare_mt.py example/ted.ref.eng example/ted.sys1.eng example/ted.sys2.eng
