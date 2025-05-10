nvcc DarpanGaur2.cu
# g++ sequential.cpp
# nvcc test.cu
# print all files in 'Public Testcases' directory using echo and for loop
echo "Public Testcases"
for file in public\ testcases/*
do
    for file2 in "$file"/*
    do
        # if file2 contains input print it
        if [[ $file2 == *"input"* ]]; then
            ./a.out "$file2" output.txt
            # find input file number, like input1.txt so 1
            fileNumber=$(echo "$file2" | grep -o '[0-9]\+')
            # echo "Testcase $fileNumber"
            # compare output.txt and output1.txt
            # if they are same print "Testcase $fileNumber passed"
            # else print "Testcase $fileNumber failed"
            if diff -s --ignore-trailing-space output.txt "$file/output$fileNumber.txt"; then
                echo "Testcase $fileNumber passed"
            else
                echo "Testcase $fileNumber failed"
            fi
        fi
    done
done

echo "Private Testcases"
for file in private\ testcases/*
do 
    if [[ $file == *"input"* ]]; then
        ./a.out "$file" output.txt
        fileNumber=$(echo "$file" | grep -o '[0-9]\+')
        if diff -s --ignore-trailing-space output.txt "private testcases/output$fileNumber.txt"; then
            echo "Testcase $fileNumber passed"
        else
            echo "Testcase $fileNumber failed"
        fi
    fi
done
