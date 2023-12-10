Make an input file which contains n and m as space seperated integers

Change the path/name of the input file in code as per your convinience.
Default path/name of input file is "inp-params.txt"

Compile :-  g++ DAM1-CO21BTECH11004.cpp
Execute :- ./a.out

Compile :- g++ SAM1-CO21BTECH11004.cpp
Execute :- ./a.out

Output file :- Primes-DAM1.txt, Primes-SAM1.txt, Times.txt

Primes-DAM1.txt :- Contains all the primes between 1-10^n using DAM1
Primes-SAM1.txt :- Contains all the primes between 1-10^n using SAM1
Times.txt :- Contains the time taken by DAM1 or SAM1 to compute all the primes between 1-10^n
            --> As the file is opened in write mode time of either SAM1 or DAM1 wolud be there after respective execution.

Time in output file is printed in microseconds with refrence to start time fetched at staring of program globally.