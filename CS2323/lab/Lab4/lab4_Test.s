# .section .data             # (instead of .data in Ripes)
.data
L1: .word 100000 #delay count to be loaded from memory

# .section .text
.text
.global main   # (add this to indicate main is a global function, need not be there in Ripes)
main:                         
la x3, L1          #this will load the ADDRESS of the data section in x3

#YOUR CODE FOLLOWS HERE. The ADDRESS of the data segment is available in x3

li x5, 0x10012004
li x6, 0x10012008
li x7, 0x1001200C

add x28, x0, x0
sw x28, 0(x5)

addi x28, x0, 0x20
sw x28, 0(x6)

#At the end, have a while(1) loop, as shown below
Lwhile1: j Lwhile1
