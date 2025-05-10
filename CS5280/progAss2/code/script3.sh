g++ BTO-CO21BTECH11004.cpp -o BTO-CO21BTECH11004
g++ MVTO-CO21BTECH11004.cpp -o MVTO-CO21BTECH11004
g++ MVTO-gc-CO21BTECH11004.cpp -o MVTO-gc-CO21BTECH11004
g++ k-MVTO-CO21BTECH11004.cpp -o k-MVTO-CO21BTECH11004

outputFile="oThreads.txt"
echo "totalDelay AbortCount" > $outputFile

for i in {1..5}
do
    echo "$((2**i)) 1000 1000 100 20 0.7" > inp-params.txt
    # print input parameters using cat
    cat inp-params.txt
    echo "Threads = $((2**i)) BTO"
    echo "Threads = $((2**i)) BTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of BTO-CO21BTECH11004 to output.txt
        ./BTO-CO21BTECH11004 >> $outputFile
    done
    echo "Threads = $((2**i)) MVTO"
    echo "Threads = $((2**i)) MVTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of MVTO-CO21BTECH11004 to output.txt
        ./MVTO-CO21BTECH11004 >> $outputFile
    done
    echo "Threads = $((2**i)) MVTO-gc"
    echo "Threads = $((2**i)) MVTO-gc" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of MVTO-gc-CO21BTECH11004 to output.txt
        ./MVTO-gc-CO21BTECH11004 >> $outputFile
    done
    echo "Threads = $((2**i)) k-MVTO"
    echo "Threads = $((2**i)) k-MVTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of k-MVTO-CO21BTECH11004 to output.txt
        ./k-MVTO-CO21BTECH11004 >> $outputFile
    done

done
