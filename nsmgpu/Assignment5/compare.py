import numpy as np
import sys
 
testCaseNum = sys.argv[1]
testcase_i = "TC" + str(testCaseNum)


out_actual = np.loadtxt('Outputs_Actual/' + testcase_i + '.txt', dtype=float)
out_expected = np.loadtxt('Outputs_Expected/' + testcase_i + '.txt', dtype=float)

diff = np.absolute(np.subtract(out_actual, out_expected))

if np.max(diff) < 1e-3:
    print(testcase_i + " Passed")
else:
    print(testcase_i + " Failed")