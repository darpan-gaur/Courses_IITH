g++ co21btech11004.cpp 

# print edit distance between pairs of file in input-Prog-HW2 directory
for file1 in inputs-Prog-HW2/*; do
    for file2 in inputs-Prog-HW2/*; do
        if [ "$file1" != "$file2" ]; then
            echo "Edit distance between $file1 and $file2:"
            ./a.out "$file1" "$file2"
        fi
    done
done