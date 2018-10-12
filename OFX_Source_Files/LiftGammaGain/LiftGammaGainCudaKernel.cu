__global__ void LiftGammaGainAdjustKernel(int p_Width, int p_Height, float p_GainL, float p_GainG, float p_GainGG, float p_GainO, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;
							
       p_Output[index + 0] = pow((p_Input[index + 0] * p_GainGG) + (p_GainL * (1.0 - (p_Input[index + 0] * p_GainGG))) + p_GainO, 1.0/p_GainG);
       p_Output[index + 1] = pow((p_Input[index + 1] * p_GainGG) + (p_GainL * (1.0 - (p_Input[index + 1] * p_GainGG))) + p_GainO, 1.0/p_GainG);
       p_Output[index + 2] = pow((p_Input[index + 2] * p_GainGG) + (p_GainL * (1.0 - (p_Input[index + 2] * p_GainGG))) + p_GainO, 1.0/p_GainG);
       p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    LiftGammaGainAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_Gain[0], p_Gain[1], p_Gain[2], p_Gain[3], p_Input, p_Output);
}
