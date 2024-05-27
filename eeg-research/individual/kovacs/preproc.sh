#!/bin/sh

python3 -u main.py -y --mode preprocess --subjects MSEL_01870 | tee ../../studentwork/kovacs/seer-msg/log/preproc_MSEL_01870.log
python3 -u main.py -y --mode preprocess --subjects MSEL_00502 | tee ../../studentwork/kovacs/seer-msg/log/preproc_MSEL_00502.log
python3 -u main.py -y --mode preprocess --subjects MSEL_01097 | tee ../../studentwork/kovacs/seer-msg/log/preproc_MSEL_01097.log
