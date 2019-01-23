head -n 1 ../example/ted.ref.eng > ref
head -n 1 ../example/ted.sys1.eng > out1
python2 RIBES.py -c -r ref out1
python2 RIBES.py -c -r ../example/ted.ref.eng ../example/ted.sys1.eng
python3 test_RIBES.py
rm ref out1
