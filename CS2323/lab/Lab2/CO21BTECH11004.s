# Name    :- Darpan Gaur
# Roll No :- CO21BTECH11004 

.data
#if needed, else ignore the data section

.text
    # instruction value is in x4 register, given in que
    # addi x4, x0, 55
    
    # get the last 7 bits -> opcode
    slli x4, x4, 57
    srli x4, x4, 57
     
    # check for R-type
    addi x28, x0, 51
    bne x4, x28, L1
    addi x10, x0, 1
    
    L1:  
     
    # check for I-type
    addi x28, x0, 19
    bne x4, x28, L2.1
    addi x10, x0, 2
    
    L2.1: 
    
    addi x28, x0, 3
    bne x4, x28, L2.2
    addi x10, x0, 2
    
    L2.2:
    
    addi x28, x0, 103
    bne x4, x28, L2.3
    addi x10, x0, 2
    
    L2.3:
    
    addi x28, x0, 115
    bne x4, x28, L2
    addi x10, x0, 2
    
    L2: 
     
    # check for B-type
    addi x28, x0, 99
    bne x4, x28, L3
    addi x10, x0, 3
    
    L3: 
    
    # check for S-type
    addi x28, x0, 35
    bne x4, x28, L4
    addi x10, x0, 4
    
    L4: 
    
    # check for J-type
    addi x28, x0, 111
    bne x4, x28, L5
    addi x10, x0, 5
    
    L5: 
    
    # check for U-type
    addi x28, x0, 55
    bne x4, x28, L6.1
    addi x10, x0, 6
    
    L6.1:
    
    addi x28, x0, 23
    bne x4, x28, L6
    addi x10, x0, 6
    
    L6: add x28, x0, x0

    #The final result should be in register x10