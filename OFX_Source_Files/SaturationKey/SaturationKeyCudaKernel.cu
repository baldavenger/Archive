__global__ void SaturationKeyAdjustKernel(int p_Width, int p_Height, float p_SatR, float p_SatG, float p_SatB, float p_SatA,
	float p_SatD, float p_SatE, float p_SatO, float p_SatZ, float p_SwitchA,
	float p_SwitchB, const float* p_Input, float* p_Output)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < p_Width) && (y < p_Height))
	{
		const int index = ((y * p_Width) + x) * 4;

		float Mx = max(p_Input[index + 0], max(p_Input[index + 1], p_Input[index + 2]));
		float mn = min(p_Input[index + 0], min(p_Input[index + 1], p_Input[index + 2]));
		float C = Mx - mn;

		float Ls = 0.5f * (Mx + mn);

		float Ss = C == 0.0f ? 0.0f : C / (1.0f - (2.0f * Ls - 1.0f));

		float r = p_SatR;
		float g = p_SatG;
		float b = p_SatB;
		float a = p_SatA;
		float d = 1.0f / p_SatD;
		float e = 1.0f / p_SatE;
		float o = p_SatO;
		float z = p_SatZ;

		float Sss = Ss == 0.0f ? 0.0f : Ss + o;
		float S = Sss < 0.0f ? 0.0f : (Sss > 1.0f ? 1.0f : Sss);
		float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= S ? 1.0f : (r >= S ? powf((r - S) / (1.0f - g), d) : 0.0f));
		float k = a == 1.0f ? 0.0f : (a + b <= S ? 1.0f : (a <= S ? powf((S - a) / b, e) : 0));
		float alpha = k * w;
		float alphaM = alpha + (1.0f - alpha) * z;
		float alphaV = (p_SwitchB == 1.0f) ? 1.0f - alphaM : alphaM;

		p_Output[index + 0] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 0];
		p_Output[index + 1] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 1];
		p_Output[index + 2] = (p_SwitchA == 1.0f) ? alphaV : p_Input[index + 2];
		p_Output[index + 3] = (p_SwitchA == 1.0f) ? p_Input[index + 3] : alphaV;
	}
}

void RunCudaKernel(int p_Width, int p_Height, float* p_Sat, float* p_Switch, const float* p_Input, float* p_Output)
{
	dim3 threads(128, 1, 1);
	dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

	SaturationKeyAdjustKernel << <blocks, threads >> >(p_Width, p_Height, p_Sat[0], p_Sat[1], p_Sat[2], p_Sat[3], p_Sat[4],
		p_Sat[5], p_Sat[6], p_Sat[7], p_Switch[0], p_Switch[1], p_Input, p_Output);
}
