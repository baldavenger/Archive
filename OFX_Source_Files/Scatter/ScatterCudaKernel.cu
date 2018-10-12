__global__ void ScatterAdjustKernel(int p_Width, int p_Height, int p_Range, float p_Mix, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;
      
	  float rg = p_Range + 1;
	 
	  int totA = round((p_Input[index + 0] + p_Input[index + 1] + p_Input[index + 2]) * 1111) + x;
	  int totB = round((p_Input[index + 0] + p_Input[index + 1]) * 1111) + y;
	 
	  int polarityA = fmodf(totA, 2) > 0.0f ? -1.0f : 1.0f;
	  int polarityB = fmodf(totB, 2) > 0.0f ? -1.0f : 1.0f;
	  int scatterA = fmodf(totA, rg) * polarityA;
	  int scatterB = fmodf(totB, rg) * polarityB;

	  int X = (x + scatterA) < 0 ? abs(x + scatterA) : ((x + scatterA) > (p_Width - 1) ? (2 * (p_Width - 1)) - (x + scatterA) : (x + scatterA));
	  int Y = (y + scatterB) < 0 ? abs(y + scatterB) : ((y + scatterB) > (p_Height - 1) ? (2 * (p_Height - 1)) - (y + scatterB) : (y + scatterB));
																													
	  p_Output[index + 0] = p_Input[((Y * p_Width) + X) * 4 + 0] * (1.0f - p_Mix) + p_Mix * p_Input[index + 0];
	  p_Output[index + 1] = p_Input[((Y * p_Width) + X) * 4 + 1] * (1.0f - p_Mix) + p_Mix * p_Input[index + 1];
	  p_Output[index + 2] = p_Input[((Y * p_Width) + X) * 4 + 2] * (1.0f - p_Mix) + p_Mix * p_Input[index + 2];
	  p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(int p_Width, int p_Height, int p_Range, float p_Mix, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    ScatterAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_Range, p_Mix, p_Input, p_Output);
}
