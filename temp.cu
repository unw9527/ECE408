#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void SpMV_JDS_T(int num_rows, float *data, int *col_index, int *jds_t_col_ptr, int jds_row_index, float *x, float *y){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < num_rows){
        float dot = 0;
        unsigned int sec = 0;
        while(jds_t_col_ptr[sec+1] - jds_t_col_ptr[sec] > row){
            dot += data[row + jds_t_col_ptr[sec]] * x[col_index[row + jds_t_col_ptr[sec]]];
            sec++;
        }
        
    }
}