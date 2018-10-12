__global__ void BalanceAdjustKernel(int p_Width, int p_Height, float p_BalR, float p_BalB, float p_LogBalR, 
	float p_LogBalB, float p_WhiteA, float p_WhiteB, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;

       p_Output[index + 0] = (p_WhiteA==1.0f) && (p_WhiteB==1.0f) ? p_Input[index + 0] + p_LogBalR : ((p_WhiteA==1.0f) ? p_Input[index + 0] * p_BalR : p_Input[index + 0]);
       p_Output[index + 1] = p_Input[index + 1];
       p_Output[index + 2] = (p_WhiteA==1.0f) && (p_WhiteB==1.0f) ? p_Input[index + 2] + p_LogBalB : ((p_WhiteA==1.0f) ? p_Input[index + 2] * p_BalB : p_Input[index + 2]);
       p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* p_Bal, float* p_LogBal, 
	float* p_White, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    BalanceAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_Bal[0], p_Bal[1], p_LogBal[0], 
		p_LogBal[1], p_White[0], p_White[1],  p_Input, p_Output);

}
