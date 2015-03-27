__kernel void multiplication(__global __write_only int* C,__global __read_only int* A,__global __read_only int* B,int size)
{
	// Calculate the row index
	int row = get_global_id(1);
	// Calculate the column index
	int col = get_global_id(0);


	int value = 0;
	for (int k = 0; k < size; ++k)
	{
		value += A[row * size + k] * B[k * size + col];
	}
	C[row * size + col] = value;
}
