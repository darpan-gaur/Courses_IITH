#include <stdio.h>
#include <cuda.h>


int main() {
    dim3 dimGrid;
    dim3 dimBlock;
    printf("dimGrid.x = %d dimGrid.y = %d dimGrid.z = %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("dimBlock.x = %d dimBlock.y = %d dimBlock.z = %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    dimGrid.x = 2;
    dimBlock.x = 2;
    printf("dimGrid.x = %d dimGrid.y = %d dimGrid.z = %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("dimBlock.x = %d dimBlock.y = %d dimBlock.z = %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    return 0;
}