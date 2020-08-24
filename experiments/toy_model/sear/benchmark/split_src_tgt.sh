#! /bin/bash
sed 's/>>.*//g' benchmark.txt > src-benchmark.txt
sed 's/.*>>//g' benchmark.txt > tgt-benchmark.txt

