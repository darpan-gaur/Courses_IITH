#.section .data # (instead of .data in Ripes it should be used in freedom studio)

.data 
L1: .word 100003 #delay count to be loaded from memory
L2: .word 0x10012004
L3: .word 0x10012008
L4: .word 0x1001200C

# .section .text
.text
.global main # (add this to indicate main is a global function, need not be there in Ripes)
main:
la x3, L1
#YOUR CODE FOLLOWS HERE. The address of the data segment is available in x3 

#The 32-bit word at address 0x10012004 should be written with 0x00000000 once atbeginning - this tells the system that no pin acts as input pin
#The 32-bit word at address 0x10012008 should be written with 0x00000020  this tells the system that the pin in position-5 from LSB acts as output

la x5, L2  
la x6, L3
la x7, L4

lw x12 0(x5)
lw x13 0(x6) 
lw x14 0(x7) 

addi x11 x0 0
sw x11 0(x12)

addi x11 x0 0x20
sw x11 0(x13)

for:
addi x21 x21 10  #by changing the value of immediate we can change the rate of delay
  
addi x11 x0 0x00
sw x11 0(x14)

last: 
addi x21 x21 -1
la x3, L1
bge x21 x0 last

addi x11 x0 0x20
sw x11 0(x14)
beq x0 x0 for

xori x7, x8, 255

 0ff44393 