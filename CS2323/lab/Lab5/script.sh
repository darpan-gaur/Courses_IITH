#! /bin/bash

g++ part1.cpp -o part1
g++ part2.cpp -o part2
for file in testcases/*
do 
    # print input file name without testcases/ prefix
    echo "Input file: - ${file:10}"
    cat $file && echo 
    echo 
    ./part1 $file
    echo
    ./part2 $file
    echo
done 