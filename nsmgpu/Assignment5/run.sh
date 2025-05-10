#!/bin/bash
# source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
# /bin/echo "##" $(whoami) is compiling mallocShared.cpp
# file=$1
# icpx -fsycl $file.cpp -o $file
# if [ $? -eq 0 ]; then ./$file input/TC1.txt; fi
# diff -s --ignore-trailing-space output.txt output/TC1.txt

# ./$file input/TC1.txt 
# diff -s --ignore-trailing-space output.txt output/TC1.txt

# TestCases="1 2 3 4 5 6 7 8"
# mkdir Outputs_Actual

# print all file with .cpp 
for file in *.cpp
do
    echo "Compiling $file"
    # icpx -fsycl /re-trailing-space output.txt output/TC1.txt
done


# TestCases="6 7 8"
# mkdir Outputs_Actual

# for i in $TestCases
# do
#     # ./$file input/TC$i.txt Outputs_Actual/TC$i.txt
#     # python compare.py $i
#     # run above 5 times
#     echo "Running TC$i"
#     for j in 1 2 3 4 5
#     do
#         ./$file input/TC$i.txt Outputs_Actual/TC$i.txt
#         python compare.py $i
#     done
# done