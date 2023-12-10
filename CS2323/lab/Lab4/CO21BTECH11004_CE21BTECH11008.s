#.section .data # (instead of .data in Ripes it should be used in freedom studio)

.data
L1: .word 100003 #delay count to be loaded from memory

# .section .text
.text
.global main # (add this to indicate main is a global function, need not be there in Ripes)
main:
la x3, L1
#YOUR CODE FOLLOWS HERE. The address of the data segment is available in x3

#The 32-bit word at address 0x10012004 should be written with 0x00000000 once atbeginning - this tells the system that no pin acts as input pin
#The 32-bit word at address 0x10012008 should be written with 0x00000020  this tells the system that the pin in position-5 from LSB acts as output

li x5, 0x10012004
li x6, 0x10012008
li x7, 0x1001200C

# change x30 register for delay
li x30, 0x00100000

add x28, x0, x0
sw x28, 0(x5)

addi x28, x0, 0x20
sw x28, 0(x6)

sw x0, 0(x7)

for:
add x21, x0, x30

add x11, x0, x0
sw x11, 0(x7)

last:
addi x21, x21, -1
la x3, L1
bge x21, x0, last

add x21, x0, x30
sw x28, 0(x7)

last2:
addi x21, x21, -1
la x3, L1
bge x21, x0, last2

sw x28, 0(x7)
beq x0, x0, for


Lwhile1: j Lwhile1