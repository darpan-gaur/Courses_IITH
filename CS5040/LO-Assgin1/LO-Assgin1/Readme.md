## Linear Optimization: Assignment 1
Implementing the simplex method for linear optimization problems.

#### Assumptions over matrix A:
1. Rank of A is N.
2. Polytope is non-degenerate.
3. Polytope is bounded

#### Input:
Input: CSV file with m+2 rows and n+1 column.
- The first row excluding the last element is the initial feasible point z of length n
- The second row excluding the last element is the cost vector c of length n
- The last column excluding the top two elements is the constraint vector b of length m
- Rows third to m+2 and column one to n is the matrix A of size m*n


#### Group Members:
- Darpan Gaur        **_CO21BTECH11004_**
- Aditya Bacharwar     **_ES21BTECH11003_**
- Bapatu Manoj Kumar Reddy     **_ES21BTECH11010_**