g++ BTO-CO21BTECH11004.cpp -o BTO-CO21BTECH11004
g++ MVTO-CO21BTECH11004.cpp -o MVTO-CO21BTECH11004
g++ MVTO-gc-CO21BTECH11004.cpp -o MVTO-gc-CO21BTECH11004
g++ k-MVTO-CO21BTECH11004.cpp -o k-MVTO-CO21BTECH11004

outputFile="oreadRatio.txt"
echo "totalDelay AbortCount" > $outputFile

# make a lsit of [0.5, 0.6, 0.7, 0.8, 0.9]
readRatioList=(0.5 0.6 0.7 0.8 0.9)

for readRatio in "${readRatioList[@]}"
do
    echo "16 1000 1000 100 20 $readRatio" > inp-params.txt
    # print input parameters using cat
    cat inp-params.txt
    echo "readRatio = $readRatio BTO"
    echo "readRatio = $readRatio BTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of BTO-CO21BTECH11004 to output.txt
        ./BTO-CO21BTECH11004 >> $outputFile
    done
    echo "readRatio = $readRatio MVTO"
    echo "readRatio = $readRatio MVTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of MVTO-CO21BTECH11004 to output.txt
        ./MVTO-CO21BTECH11004 >> $outputFile
    done
    echo "readRatio = $readRatio MVTO-gc"
    echo "readRatio = $readRatio MVTO-gc" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of MVTO-gc-CO21BTECH11004 to output.txt
        ./MVTO-gc-CO21BTECH11004 >> $outputFile
    done
    echo "readRatio = $readRatio k-MVTO"
    echo "readRatio = $readRatio k-MVTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of k-MVTO-CO21BTECH11004 to output.txt
        ./k-MVTO-CO21BTECH11004 >> $outputFile
    done

done
