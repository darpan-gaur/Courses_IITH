.data
.dword 3, 12, 3, 125, 50, 32, 16

.text
    lui x3, 0x10000
    addi x3, x3, 0x100
    ld x10, 0(x3)
    lui x5, 0x10000
    addi x5, x5, 0x100
for:
   addi x3, x3, 8
   ld x12, 0(x3)
   addi x3, x3, 8
   ld x13, 0(x3)
   
   gcd:
       beq x12, x13, exit
       bge x13, x12, else
           sub x12, x12, x13
           beq x0, x0 , else2
       
       else:
           sub x13, x13, x12
       else2:
       
       beq x0, x0 gcd
       exit:
       
       sd x12, 0(x5)
       addi x5, x5, 8
       
   addi x10, x10, -1
   
blt x0, x10 ,for
         
    