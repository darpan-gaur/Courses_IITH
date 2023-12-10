g++ CLH-CO21BTECH11004.cpp -o CLH
g++ MCS-CO21BTECH11004.cpp -o MCS
#g++ MCS.cpp -o MCS
echo "n, time" > out.txt
for i in {1..6}
do
    n=$((2**i))
    echo $n 15 1 2 > inp-params.txt
    ./CLH
    for j in {1..5}
    do
        echo "CLH $n " 
        echo -n "$n " >> out.txt
        ./CLH >> out.txt
    done
    echo "" >> out.txt
    # echo $n 15 5 20 
    ./MCS
    for j in {1..5}
    do 
        echo "MCS $n "
        echo -n "$n " >> out.txt
        ./MCS >> out.txt
        # ./MCS
    done
    echo "" >> out.txt
done

echo "n, time" > out1.txt
for i in {1..5}
do 
    k=$((5*i))
    echo 16 $k 1 2 > inp-params.txt
    ./CLH
    for j in {1..5}
    do
        echo "CLH $k " 
        echo -n "$k " >> out1.txt
        ./CLH >> out1.txt
    done
    echo "" >> out1.txt
    # echo $n 15 5 20
    ./MCS
    for j in {1..5}
    do 
        echo "MCS $k "
        echo -n "$k " >> out1.txt
        ./MCS >> out1.txt
        # ./MCS
    done
    echo "" >> out1.txt
done
