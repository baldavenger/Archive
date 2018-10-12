__global__ void LumaKeyAdjustKernel(int p_Width, int p_Height, float p_LumaKeyR, float p_LumaKeyG, float p_LumaKeyB, float p_LumaKeyA,
 float p_LumaKeyD, float p_LumaKeyE, float p_LumaKeyO, float p_LumaKeyZ, float p_SwitchA, float p_SwitchB, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;

     float l = (p_Input[index + 0] * 0.2126f) + (p_Input[index + 1] * 0.7152f) + (p_Input[index + 2] * 0.0722f);  
	 float L = l - p_LumaKeyO;								
     float q = fmin(L, 1.0f);									
     float n = fmax(q, 0.0f);											
       																 
     float r = p_LumaKeyR;				
     float g = p_LumaKeyG;				
     float b = p_LumaKeyB;				
     float a = p_LumaKeyA;				
     float d = 1.0f / p_LumaKeyD;							
     float e = 1.0f / p_LumaKeyE;							
     float z = p_LumaKeyZ;						 
		
     float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= n ? 1.0f : (r >= n ? powf((r - n) / (1.0f - g), d) : 0.0f));		
     float k = a == 1.0f ? 0.0f : (a + b <= n ? 1.0f : (a <= n ? powf((n - a) / b, e) : 0.0f));						
     float alpha = k * w;									
     float alphaM = alpha + (1.0f - alpha) * z;		 
	 float alphaV = (p_SwitchB == 1.0f) ? 1.0 - alphaM : alphaM;	
       																												
	 p_Output[index + 0] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 0];
	 p_Output[index + 1] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 1];
	 p_Output[index + 2] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 2];
	 p_Output[index + 3] = (p_SwitchA == 1.0f) ? p_Input[index + 3] : alphaV;
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* p_LumaKey, float* p_Switch, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    LumaKeyAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_LumaKey[0], p_LumaKey[1], p_LumaKey[2], p_LumaKey[3], p_LumaKey[4],
     p_LumaKey[5], p_LumaKey[6], p_LumaKey[7], p_Switch[0], p_Switch[1], p_Input, p_Output);
}
