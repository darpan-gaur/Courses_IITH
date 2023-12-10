# Lab1 Basic Assembly Programming
# Name :-         Darpan Gaur
# Roll Number :-  CO21BTECH11004

.data
#The following line defines the 15 values present in the memory.
.dword 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 523, 524, 525, 533, 512


.text
    #The following line initializes register x3 with 0x10000000 
    #so that you can use x3 for referencing various memory locations. 
    lui x3, 0x10000
    
    # initilaize x10 with zero
    add x10, x0, x0
    
    # add first 10 numbers
    ld x5, 0(x3)
    add x10, x10, x5
    
    ld x5, 8(x3)
    add x10, x10, x5
    
    ld x5, 16(x3)
    add x10, x10, x5
    
    ld x5, 24(x3)
    add x10, x10, x5
    
    ld x5, 32(x3)
    add x10, x10, x5
    
    ld x5, 40(x3)
    add x10, x10, x5
    
    ld x5, 48(x3)
    add x10, x10, x5
    
    ld x5, 56(x3)
    add x10, x10, x5
    
    ld x5, 64(x3)
    add x10, x10, x5
    
    ld x5, 72(x3)
    add x10, x10, x5
    
    # subtract next 5 numbers
    ld x5, 80(x3)
    sub x10, x10, x5
    
    ld x5, 88(x3)
    sub x10, x10, x5
    
    ld x5, 96(x3)
    sub x10, x10, x5
    
    ld x5, 104(x3)
    sub x10, x10, x5
    
    ld x5, 112(x3)
    sub x10, x10, x5
 
    #The final result should be in register x10
