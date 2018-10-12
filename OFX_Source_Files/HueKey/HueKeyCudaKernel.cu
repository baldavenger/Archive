__global__ void HueKeyAdjustKernel(int p_Width, int p_Height, float p_HueR, float p_HueG, float p_HueB, float p_HueA,
 float p_HueD, float p_HueE, float p_HueO, float p_HueZ, float p_SwitchA, float p_SwitchB, const float* p_Input, float* p_Output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;
      
    float Mx = max(p_Input[index + 0], max(p_Input[index + 1], p_Input[index + 2]));				
    float mn = min(p_Input[index + 0], min(p_Input[index + 1], p_Input[index + 2]));	 
	float del_Max = Mx - mn;
	
	float del_R = (((Mx - p_Input[index + 0]) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    float del_G = (((Mx - p_Input[index + 1]) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    float del_B = (((Mx - p_Input[index + 2]) / 6.0f) + (del_Max / 2.0f)) / del_Max;
   
    float h = del_Max == 0.0f ? 0.0f : (p_Input[index + 0] == Mx ? del_B - del_G : (p_Input[index + 1] == Mx ? (1.0f / 3.0f) + del_R - del_B : (2.0f / 3.0f) + del_G - del_R));
       														
	float r = p_HueR;				
	float g = p_HueG;					
	float b = p_HueB;					
	float a = p_HueA;					
	float d = 1.0f / p_HueD;							
	float e = 1.0f / p_HueE;							
	float o = p_HueO * -1.0f;							
	float z = p_HueZ;								
																			   
	float H = h == 0.0f ? 0.0f : (h + o > 1.0f ? h + o - 1.0f : (h + o < 0.0f ? 1.0f + h + o : h + o));															
	float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= H ? 1.0f : (r >= H ? powf((r - H) / (1.0f - g), d) : 0.0f));
	float k = a == 1.0f ? 0.0f : (a + b <= H ? 1.0f : (a <= H ? powf((H - a) / b, e) : 0.0f));	
	float alpha = k * w;										
	float alphaM = alpha + (1.0f - alpha) * z;				
	float alphaV = (p_SwitchB == 1.0f) ? 1.0f - alphaM : alphaM;		
																																								   
	p_Output[index + 0] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 0];
	p_Output[index + 1] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 1];
	p_Output[index + 2] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 2];
	p_Output[index + 3] = (p_SwitchA == 1.0f) ? p_Input[index + 3] : alphaV;
   }																	   
}

void RunCudaKernel(int p_Width, int p_Height, float* p_Hue, float* p_Switch, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    HueKeyAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_Hue[0], p_Hue[1], p_Hue[2], p_Hue[3], p_Hue[4],
     p_Hue[5], p_Hue[6], p_Hue[7], p_Switch[0], p_Switch[1], p_Input, p_Output);
}
