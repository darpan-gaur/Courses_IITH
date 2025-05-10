g++ BTO-CO21BTECH11004.cpp -o BTO-CO21BTECH11004
g++ MVTO-CO21BTECH11004.cpp -o MVTO-CO21BTECH11004
g++ MVTO-gc-CO21BTECH11004.cpp -o MVTO-gc-CO21BTECH11004
g++ k-MVTO-CO21BTECH11004.cpp -o k-MVTO-CO21BTECH11004

outputFile="oVar.txt"
echo "totalDelay AbortCount" > $outputFile

for i in {1..5}
do
    echo "16 $((i*1000)) 1000 100 20 0.7" > inp-params.txt
    # print input parameters using cat
    cat inp-params.txt
    echo "Var = $((i*1000)) BTO"
    echo "Var = $((i*1000)) BTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of BTO-CO21BTECH11004 to output.txt
        ./BTO-CO21BTECH11004 >> $outputFile
    done
    echo "Var = $((i*1000)) MVTO"
    echo "Var = $((i*1000)) MVTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of MVTO-CO21BTECH11004 to output.txt
        ./MVTO-CO21BTECH11004 >> $outputFile
    done
    echo "Var = $((i*1000)) MVTO-gc"
    echo "Var = $((i*1000)) MVTO-gc" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of MVTO-gc-CO21BTECH11004 to output.txt
        ./MVTO-gc-CO21BTECH11004 >> $outputFile
    done
    echo "Var = $((i*1000)) k-MVTO"
    echo "Var = $((i*1000)) k-MVTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of k-MVTO-CO21BTECH11004 to output.txt
        ./k-MVTO-CO21BTECH11004 >> $outputFile
    done

done
