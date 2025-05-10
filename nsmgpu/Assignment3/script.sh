nvcc DarpanGaur.cu -arch=sm_70 -rdc=true

echo "Public Testcases"
for file in Public\ Test\ cases/*.txt
do 
    # if file name contains "input"
    if [[ $file == *"input"* ]]; then
        # run ./a.out $file and wait for it to finish
        # echo $file
        ./a.out $file
        wait
        # diff outputN.txt "${file/input/output}"
        python3 compare.py outputN.txt "${file/input/output}"
    fi
done

echo "Private Testcases"
for file in Private\ testcases/*.txt
do 
    # if file name contains "input"
    if [[ $file == *"input"* ]]; then
        # echo "Running $file"
        # echo $file
        ./a.out $file
        wait
        # diff outputN.txt "${file/input/output}"
        python3 compare.py outputN.txt "${file/input/output}"
    fi
done