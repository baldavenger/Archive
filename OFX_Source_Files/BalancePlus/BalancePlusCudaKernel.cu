__global__ void BalancePlusAdjustKernel(int p_Width, int p_Height, float balGainR, float balGainB, float balOffsetR, float balOffsetB,
float balLiftR, float balLiftB, float lumaMath, float lumaLimit, float GainBalance, float OffsetBalance, float WhiteBalance, 
float PreserveLuma, float DisplayAlpha, float LumaRec, float LumaAvg, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;

	float lumaRec = p_Input[index + 0] * 0.2126f + p_Input[index + 1] * 0.7152f + p_Input[index + 2] * 0.0722f;
	float lumaAvg = (p_Input[index + 0] + p_Input[index + 1] + p_Input[index + 2]) / 3.0f;
	float lumaMax = fmax(fmax(p_Input[index + 0], p_Input[index + 1]), p_Input[index + 2]);
	float luma = LumaRec == 1.0f ? lumaRec : LumaAvg == 1.0f ? lumaAvg : lumaMax;

	float alpha = lumaLimit > 1.0f ? luma + (1.0f - lumaLimit) * (1.0f - luma) : lumaLimit >= 0.0f ? (luma >= lumaLimit ? 
	1.0f : luma / lumaLimit) : lumaLimit < -1.0f ? (1.0f - luma) + (lumaLimit + 1.0f) * luma : luma <= (1.0f + lumaLimit) ? 1.0f : 
	(1.0f - luma) / (1.0f - (lumaLimit + 1.0f));
	float Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha;

	float BalR = GainBalance == 1.0f ? p_Input[index + 0] * balGainR : OffsetBalance == 1.0f ? p_Input[index + 0] + balOffsetR : p_Input[index + 0] + (balLiftR * (1.0f - p_Input[index + 0]));
	float BalB = GainBalance == 1.0f ? p_Input[index + 2] * balGainB : OffsetBalance == 1.0f ? p_Input[index + 2] + balOffsetB : p_Input[index + 2] + (balLiftB * (1.0f - p_Input[index + 2]));
	float Red = WhiteBalance == 1.0f ? ( PreserveLuma == 1.0f ? BalR * lumaMath : BalR) : p_Input[index + 0];
	float Green = WhiteBalance == 1.0f && PreserveLuma == 1.0f ? p_Input[index + 1] * lumaMath : p_Input[index + 1];
	float Blue = WhiteBalance == 1.0f ? ( PreserveLuma == 1.0f ? BalB * lumaMath : BalB) : p_Input[index + 2];

	p_Output[index + 0] = DisplayAlpha == 1.0f ? Alpha : Red * Alpha + p_Input[index + 0] * (1.0f - Alpha);
	p_Output[index + 1] = DisplayAlpha == 1.0f ? Alpha : Green * Alpha + p_Input[index + 1] * (1.0f - Alpha);
	p_Output[index + 2] = DisplayAlpha == 1.0f ? Alpha : Blue * Alpha + p_Input[index + 2] * (1.0f - Alpha);
	p_Output[index + 3] = DisplayAlpha == 1.0f ? p_Input[index + 3] : Alpha;
	
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* balGain, float* balOffset, 
float* balLift, float* lumaMath, float* lumaLimit, float* GainBalance, float* OffsetBalance, float* WhiteBalance, 
float* PreserveLuma, float* DisplayAlpha, float* LumaRec, float* LumaAvg, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    BalancePlusAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, balGain[0], balGain[1], balOffset[0], balOffset[1],
     balLift[0], balLift[1], lumaMath[0], lumaLimit[0], GainBalance[0], OffsetBalance[0], WhiteBalance[0], 
     PreserveLuma[0], DisplayAlpha[0], LumaRec[0], LumaAvg[0],  p_Input, p_Output);

}
