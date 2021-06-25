#! /bin/bash
DIR=$1
sed 's/>>.*//g' $DIR/train.txt > $DIR/src-train.txt
sed 's/>>.*//g' $DIR/val.txt > $DIR/src-val.txt
sed 's/>>.*//g' $DIR/test.txt > $DIR/src-test.txt

sed 's/.*>>//g' $DIR/train.txt > $DIR/tgt-train.txt
sed 's/.*>>//g' $DIR/val.txt > $DIR/tgt-val.txt
sed 's/.*>>//g' $DIR/test.txt > $DIR/tgt-test.txt

# also randomly shuffles training set
paste -d ':' $DIR/src-train.txt $DIR/tgt-train.txt | sort -R > $DIR/train.txt
sed 's/.*://g' $DIR/train.txt > $DIR/tgt-train.txt
sed 's/:.*//g' $DIR/train.txt > $DIR/src-train.txt
