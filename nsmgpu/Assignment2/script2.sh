g++ seq2.cpp

for i in {1..10}
do
    ./a.out
    # copy input.txt  and output.txt 
    cp input.txt "public testcases/myTestcase/input$i.txt"
    cp output.txt "public testcases/myTestcase/output$i.txt"
done