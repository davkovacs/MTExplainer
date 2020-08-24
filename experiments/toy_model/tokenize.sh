#! /bin/bash
big_dir=./data_big2/
sed 's/>>.*//g' train_big2.txt > ${big_dir}src-train.txt
sed 's/>>.*//g' val_big2.txt > ${big_dir}src-val.txt
sed 's/>>.*//g' test_big2.txt > ${big_dir}src-test.txt

sed 's/.*>>//g' train_big2.txt > ${big_dir}tgt-train.txt
sed 's/.*>>//g' val_big2.txt > ${big_dir}tgt-val.txt
sed 's/.*>>//g' test_big2.txt > ${big_dir}tgt-test.txt

python tokenize_rxns.py ${big_dir}src-train.txt
python tokenize_rxns.py ${big_dir}tgt-train.txt
python tokenize_rxns.py ${big_dir}src-val.txt
python tokenize_rxns.py ${big_dir}tgt-val.txt
python tokenize_rxns.py ${big_dir}src-test.txt
python tokenize_rxns.py ${big_dir}tgt-test.txt

rm train_big.txt
rm val_big.txt
rm test_big.txt
