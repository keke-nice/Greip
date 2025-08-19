#!/bin/bash
python train.py \
-sr 0.4 \
-cr 0 -cs 200 \
-gr 0 -gm1 0 -gm2 0 -gart 0.4 \
-lr 0.3 -la 0.8 \
-qr 0 -qs 100 \
-tr 0.3 -tl 30 \
-mi 40000 \
-t 'V4V_new' \
-b 64 \
-aa 1 \
-bb 1 \
-cc 1




