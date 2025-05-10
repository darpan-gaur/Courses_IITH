nvcc DarpanGaur.cu

for file in Public\ testcases/*
do
    if [[ $file == *"input"* ]]; then
        # echo "Running $file"
        ./a.out "$file" output.txt
        flieNumber=$(echo $file | grep -o '[0-9]\+')
        # echo $flieNumber
        if diff -s --ignore-trailing-space output.txt "Public testcases/output$flieNumber.txt" ; then
            echo "Testcase $flieNumber passed"
        else
            echo "Testcase $flieNumber failed"
        fi
    fi
done