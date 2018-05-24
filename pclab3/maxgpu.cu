#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define TPB 1024 //elements per thread
#define BN 64    //block number

/*function declarations*/
int getmax(int *, int);
__global__ void kernel_getmax(int *, int, int);
__device__ void thread_getmax(int *, int *, int, int);

//the sequential version of getmax
int getmax(int num[], int size){
    int i;
    int max=num[0];
    for(i=1;i<size;i++){
	if(num[i]>max){
	    max=num[i];
	}
    }
    return max;
}

/*kernel called by the host to getmax. 
  The high level idea is that each thread first find the max in its share of TPB elements. 
  Then each block use the reduction tree algorithm to find the max in the block. 
  After the two steps above, the max can be found among the first BN elements of the array num[]. 
  We move them back to CPU and use the sequential version to find the max among the last BN elements.*/
__global__ void kernel_getmax(int num[], int size, int workload)
{
    //first, we ask each thread to find the max in its assigned EPT random numbers
    __shared__ int max_each_thread[TPB]; 
    thread_getmax(num, max_each_thread, size, workload);
    __syncthreads();
    
    //next, we find the max in a block. note that the same tree algorithm for parallel summation applies to max as well
    int thread_id=threadIdx.x;
    //loop unrolling for efficiency
    if(thread_id<512){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+512]){
            max_each_thread[thread_id]=max_each_thread[thread_id+512];
        }
    }
    __syncthreads();
    if(thread_id<256){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+256]){
            max_each_thread[thread_id]=max_each_thread[thread_id+256];
        }
    }
    __syncthreads();
    if(thread_id<128){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+128]){
            max_each_thread[thread_id]=max_each_thread[thread_id+128];
        }
    }
    __syncthreads();
    if(thread_id<64){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+64]){
            max_each_thread[thread_id]=max_each_thread[thread_id+64];
        }
    }
    __syncthreads();
    if(thread_id<32){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+32]){
            max_each_thread[thread_id]=max_each_thread[thread_id+32];
        }
    }
    __syncthreads();
    if(thread_id<16){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+16]){
            max_each_thread[thread_id]=max_each_thread[thread_id+16];
        }
    }
    __syncthreads();
    if(thread_id<8){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+8]){
            max_each_thread[thread_id]=max_each_thread[thread_id+8];
        }
    }
    __syncthreads();    
    if(thread_id<4){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+4]){
            max_each_thread[thread_id]=max_each_thread[thread_id+4];
        }
    }
    __syncthreads();
    if(thread_id<2){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+2]){
            max_each_thread[thread_id]=max_each_thread[thread_id+2];
        }
    }
    __syncthreads();
    if(thread_id<1){
        if(max_each_thread[thread_id]<max_each_thread[thread_id+1]){
            max_each_thread[thread_id]=max_each_thread[thread_id+1];
        }
    }
    __syncthreads();

    //we put the max of the i-th block at num[i]
    if(thread_id==0){
	num[blockIdx.x]=max_each_thread[0];
    }
}

/*The function called by the kernel. The sequential getmax version for each thread*/
__device__ void thread_getmax(int num[], int max_each_thread[], int size, int workload){
    int max=0;
    int index=workload*(blockIdx.x*blockDim.x+threadIdx.x);
    int i;
    for(i=index;(i<size)&&(i<index+workload);i++){
	    if(max<num[i]){
		max=num[i];
	    }
    }

    //store the max of this thread in the corresponding position of the shared array
    max_each_thread[threadIdx.x]=max;
}


int main(int argc, char *argv[])
{
    int size = 0;  // The size of the array
    int i;  // loop index
    int * numbers; //pointer to the array
    
    if(argc !=2)
    {
       printf("usage: maxgpu num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    numbers = (int *)malloc(size * sizeof(int));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < size; i++)
       numbers[i] = rand()  % size;    

    /*todo: 1)allocate memory and copy numbers from host to device 
	    2)invoke kernels to deal with the array
	    3)copy numbers from device to host and free memory*/
    
    int workload=ceil((double)size/(TPB*BN));
    //step1:memory setup
    int *gpu_numbers;
    cudaError_t err;
    err=cudaMalloc((void**)&gpu_numbers, sizeof(int)*size);
    //sometimes when x=100,000,000, we might fail to allocate/transfer memory
    if(err!=cudaSuccess){
	printf("Cannot allocate memory for the initial random array\n");
    }
    err=cudaMemcpy(gpu_numbers, numbers, sizeof(int)*size, cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){
	printf("cannot pass the random array from cpu to gpu\n");
    }
    
    //step2:invoke kernal
    kernel_getmax<<<BN, TPB>>>(gpu_numbers, size, workload);
    
    //step3:copy the max from device & housekeeping(free the allocated pointers)
    cudaMemcpy(numbers,gpu_numbers, sizeof(int)*BN, cudaMemcpyDeviceToHost);
    printf(" The maximum number calculated from GPU is: %d\n", getmax(numbers, BN));
    cudaFree(gpu_numbers);
    free(numbers);
    exit(0);
}


