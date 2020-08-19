#! /bin/bash

datadir=/home/dpk25/rds/hpc-work/toy_model/data_sear_small/
python process_preds.py -predictions ${datadir}preds_on_test_50k.txt \
                        -targets ${datadir}tgt-test.txt -reactions ${datadir}test_rxs.txt \
                        -outp ${datadir}mistake_rxs.txt
