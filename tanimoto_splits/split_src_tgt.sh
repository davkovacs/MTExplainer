#! /bin/bash
sed 's/>>.*//g' train.txt > src-train.txt
sed 's/>>.*//g' val.txt > src-val.txt
sed 's/>>.*//g' test.txt > src-test.txt

sed 's/.*>>//g' train.txt > tgt-train.txt
sed 's/.*>>//g' val.txt > tgt-val.txt
sed 's/.*>>//g' test.txt > tgt-test.txt

# also randomly shuffles training set
paste -d ':' src-train.txt tgt-train.txt | sort -R > train.txt
sed 's/.*://g' train.txt > tgt-train.txt
sed 's/:.*//g' train.txt > src-train.txt
