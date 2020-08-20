#! /bin/bash

data_dir=/home/dpk25/rds/hpc-work/toy_model/data_sear_small/
sed 's/ //g' ${data_dir}src-test.txt > ${data_dir}clean-src.txt
sed 's/ //g' ${data_dir}tgt-test.txt > ${data_dir}clean-tgt.txt
paste -d '>' ${data_dir}clean-src.txt ${data_dir}clean-tgt.txt | sed 's/>/>>/g' > ${data_dir}test_rxs.txt
rm ${data_dir}clean-src.txt
rm ${data_dir}clean-tgt.txt

