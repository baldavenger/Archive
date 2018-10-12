__global__ void BlueBoxAdjustKernel(int p_Width, int p_Height, float p_BlueBoxR, float p_BlueBoxG, float p_BlueBoxB, float p_BlueBoxA,
	float p_BlueBoxD, float p_BlueBoxE, float p_BlueBoxO, float p_BlueBoxZ, float p_SwitchFA, float p_SwitchFB, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;

	   float l = p_Input[index + 0] * 0.2126f + p_Input[index + 1] * 0.7152f + p_Input[index + 2] * 0.0722f;
	   float blue = p_Input[index + 2] > p_Input[index + 0] ? p_Input[index + 0] : p_Input[index + 2];  
	   float L = l - p_BlueBoxO;								
       float q = min(L, 1.0f);									
       float n = max(q, 0.0f);	
       
	   float r = p_BlueBoxR;
	   float g = p_BlueBoxG;
	   float b = p_BlueBoxB;
	   float a = p_BlueBoxA;
	   float d = 1.0f / p_BlueBoxD;
	   float e = 1.0f / p_BlueBoxE;
	   float z = p_BlueBoxZ;
	   float w = r + n == 0.0f ? 0.0f : (r - (1.0f - g) >= n ? 1.0f : (r >= n ? powf((r - n) / (1.0f - g), d) : 0.0f));
	   float k = a + n == 2.0f ? 0.0f : (a + b <= n ? 1.0f : (a <= n ? powf((n - a) / b, e) : 0.0f));
	   float alpha = k * w;
	   float alphaM = alpha + ((1.0f - alpha) * z);
	   float alphaV = (p_SwitchFB == 1.0f) ? (1.0 - alphaM) : alphaM;
	   p_Output[index + 0] = (p_SwitchFA == 1.0f) ? alphaV : p_Input[index + 0];
	   p_Output[index + 1] = (p_SwitchFA == 1.0f) ? alphaV : p_Input[index + 1];
	   p_Output[index + 2] = (p_SwitchFA == 1.0f) ? alphaV : (p_Input[index + 2] * (1.0f - alphaV)) + (blue * alphaV);
	   p_Output[index + 3] = (p_SwitchFA == 1.0f) ? p_Input[index + 3] : alphaV;
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* p_BlueBox, float* p_Switch, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    BlueBoxAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_BlueBox[0], p_BlueBox[1], p_BlueBox[2], p_BlueBox[3], p_BlueBox[4],
		p_BlueBox[5], p_BlueBox[6], p_BlueBox[7], p_Switch[0], p_Switch[1], p_Input, p_Output);
}
