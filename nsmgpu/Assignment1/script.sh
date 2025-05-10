nvcc DarpanGaur.cu
# g++ sequential.cpp
# print all files in 'Public Testcases' directory using echo and for loop
echo "Running Public Testcases"
for file in Public\ Testcases/*
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
            if diff --ignore-trailing-space output.txt "$file/output$fileNumber.txt"; then
                echo "Testcase $fileNumber passed"
            else
                echo "Testcase $fileNumber failed"
            fi
        fi
    done
done

echo "Running Private Testcases"
for file in Private\ Testcases/input/*
do  
    # echo "$file"
    ./a.out "$file" output.txt
    # find input file number, like input1.txt so 1
    fileNumber=$(echo "$file" | grep -o '[0-9]\+')
    # echo "Testcase $fileNumber"
    # compare output.txt and output1.txt
    # if they are same print "Testcase $fileNumber passed"
    # else print "Testcase $fileNumber failed"
    if diff --ignore-trailing-space output.txt "Private Testcases/output/output$fileNumber.txt"; then
        echo "Testcase $fileNumber passed"
    else
        echo "Testcase $fileNumber failed"
    fi
done
