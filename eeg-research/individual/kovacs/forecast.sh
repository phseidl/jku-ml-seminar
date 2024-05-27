#!/bin/sh

python3 -u main.py --mode forecast --subjects MSEL_00172 | tee ../../studentwork/kovacs/seer-msg/results/MSEL_00172_results.txt
python3 -u main.py --mode forecast --subjects MSEL_00764 | tee ../../studentwork/kovacs/seer-msg/results/MSEL_00764_results.txt
python3 -u main.py --mode forecast --subjects MSEL_01676 | tee ../../studentwork/kovacs/seer-msg/results/MSEL_01676_results.txt
python3 -u main.py --mode forecast --subjects MSEL_01838 | tee ../../studentwork/kovacs/seer-msg/results/MSEL_01838_results.txt
python3 -u main.py --mode forecast --subjects MSEL_01709 | tee ../../studentwork/kovacs/seer-msg/results/MSEL_01709_results.txt
