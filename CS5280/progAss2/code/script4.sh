g++ k-MVTO-CO21BTECH11004_2.cpp -o k-MVTO-CO21BTECH11004_2

outputFile="oKsize.txt"
echo "totalDelay AbortCount" > $outputFile

for i in {5..25..5}
do 
    echo "16 1000 1000 100 20 0.7 $i" > inp-params.txt
    # print input parameters using cat
    cat inp-params.txt
    echo "Ksize = $i k-MVTO"
    echo "Ksize = $i k-MVTO" >> $outputFile
    for j in {1..3}
    do
        # print system time
        echo "System time: $(date)"
        # output of k-MVTO-CO21BTECH11004_2 to output.txt
        ./k-MVTO-CO21BTECH11004_2 >> $outputFile
    done
done