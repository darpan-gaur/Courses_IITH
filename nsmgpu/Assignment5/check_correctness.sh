#!/bin/bash

filename=$1
# dpcpp $filename -o matmul
nvcc $filename -o matmul
TestCases="1 2 3 4 5 6 7 8"

mkdir Outputs_Actual

for i in $TestCases
do
    ./matmul < TestCases/TC$i.txt > Outputs_Actual/TC$i.txt
    python compare.py $i
done

rm -r Outputs_Actual

