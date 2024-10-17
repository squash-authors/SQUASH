#!/bin/bash
set -x

python sb_auto_runner.py --run_dir sift1m_bb512_p10_csuint8_30iter
echo
echo

python sb_auto_runner.py --run_dir sift1m_bb512_p20_csuint8_30iter
echo
echo

python sb_auto_runner.py --run_dir sift1m_bb512_p40_csuint8_30iter
echo
echo

python sb_auto_runner.py --run_dir sift10m_bb512_p10_csuint8_30iter
echo
echo


set +x
