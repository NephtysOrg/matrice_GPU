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
}
