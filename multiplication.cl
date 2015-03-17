<<<<<<< HEAD
__kernel void multiplication(__global __write_only int *C,__global __read_only int* A,__global __read_only int* B,const unsigned int size)
{
   int rang = get_global_id(0); 
   int rang1 = get_global_id(1);
 
   // value stores the element that is 
   // computed by the thread
   int result = 0;
   for (int k = 0; k < size; ++k)
   {
      int tmp_A = A[rang1 * size + k];
      int tmp_B = B[k * size + rang];
      result += tmp_A * tmp_B;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[rang1 * size + rang] = result;
=======
__kernel void multiplication(__global __write_only int* output,__global __read_only int* input,const unsigned int size)
{
	unsigned int index=get_global_id(0);

	for(int i=0; i<size; i++){
	int pair = i%2;
		if(index%2 != pair){
			if(index+1==size)
			{
				output[index]=input[index];
			}else{
				output[index]=min(input[index],input[index+1]);
			} 	
		}else{
			if(index>0)
			{
				output[index]=max(input[index],input[index-1]);
			}else{
				output[index]=input[index];
			}
		}
	input[index] = output[index];
	}
>>>>>>> f09a5c10e83dc8ee17b996dfcd2d0cdf37251fc1
}
