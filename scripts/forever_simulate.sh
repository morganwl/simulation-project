#!/bin/bash

trials=$1
perts=( '1,2,120' '1,4,120' '1,6,120' '1,8,120' '1,10,120' '1,12,120' '1,14,120' '1,16,120' '1,18,120' '1,20,120' '1,22,120' '1,24,120' '1,26,120' '1,28,120' '1,30,120' '1,32,120' \
    '1,34,120' '1,36,120' '1,38,120' '1,40,120' '1,42,120' '1,44,120' '1,46,120' '1,48,120' '1,50,120')
while true; do
    for pert in "${perts[@]}"; do
        fbsimulate -x b35-variable -n $trials -p "$pert"
        fbtabulate b35-variable b35-variable b35-variable -n experiment
    done
done
