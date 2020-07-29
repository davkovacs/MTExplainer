#! /bin/bash

data_dir=/home/dpk25/MolecularTransformer2/data/data/MIT_mixed_clean/
sed 's/ //g' ${data_dir}src-train.txt > ${data_dir}clean-src.txt
sed 's/ //g' ${data_dir}tgt-train.txt > ${data_dir}clean-tgt.txt
paste -d '>' ${data_dir}clean-src.txt ${data_dir}clean-tgt.txt | sed 's/>/>>/g' > clean-train.txt
rm ${data_dir}clean-src.txt
rm ${data_dir}clean-tgt.txt

