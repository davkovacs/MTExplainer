#! /bin/bash
sed 's/>>.*//g' train.txt > src-train.txt
sed 's/>>.*//g' val.txt > src-val.txt
sed 's/>>.*//g' test.txt > src-test.txt

sed 's/.*>>//g' train.txt > tgt-train.txt
sed 's/.*>>//g' val.txt > tgt-val.txt
sed 's/.*>>//g' test.txt > tgt-test.txt

python preprocess.py src-train.txt
python preprocess.py tgt-train.txt
python preprocess.py src-val.txt
python preprocess.py tgt-val.txt
python preprocess.py src-test.txt
python preprocess.py tgt-test.txt

rm train.txt
rm val.txt
rm test.txt
