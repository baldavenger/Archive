__global__ void TwelveWayAdjustKernel(int p_Width, int p_Height, float p_SwitchO, float p_SwitchS,
	 float p_SwitchM, float p_SwitchH, float p_GainOL, float p_GainOG, float p_GainOGG, 
    float p_GainSL, float p_GainSG, float p_GainSGG, float p_GainSA, float p_GainSB, 
    float p_GainML, float p_GainMG,  float p_GainMGG, float p_GainMA, float p_GainMB, 
    float p_GainHL, float p_GainHG,  float p_GainHGG, float p_GainHA, float p_GainHB,
     const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;
		
    float Ro = p_Input[index + 0] * p_GainOGG + p_GainOL * (1.0f - (p_Input[index + 0] * p_GainOGG));
    float RO = Ro >= 0.0f && Ro <= 1.0f ? (p_SwitchO != 1.0f ? pow(Ro, 1.0f / p_GainOG) : 1.0f - pow(1.0f - Ro, p_GainOG)) : Ro;
    
    float Go = p_Input[index + 1] * p_GainOGG + p_GainOL * (1.0f - (p_Input[index + 1] * p_GainOGG));
    float GO = Go >= 0.0f && Go <= 1.0f ? (p_SwitchO != 1.0f ? pow(Go, 1.0f / p_GainOG) : 1.0f - pow(1.0f - Go, p_GainOG)) : Go;
    
    float Bo = p_Input[index + 2] * p_GainOGG + p_GainOL * (1.0f - (p_Input[index + 2] * p_GainOGG));
    float BO = Bo >= 0.0f && Bo <= 1.0f ? (p_SwitchO != 1.0f ? pow(Bo, 1.0f / p_GainOG) : 1.0f - pow(1.0f - Bo, p_GainOG)) : Bo;
                       
	float Rs = (RO - p_GainSA) / (p_GainSB - p_GainSA);
	float Rss = Rs >= 0.0f && Rs <= 1.0f ? Rs * p_GainSGG + p_GainSL * (1.0f - (Rs * p_GainSGG)) : Rs;
	float RS = Rss >= 0.0f && Rss <= 1.0f ? (p_SwitchS != 1.0f ? pow(Rss, 1.0f / p_GainSG) : 1.0f - pow(1.0f - Rss, p_GainSG)) : Rss;
	float RSS = RS * (p_GainSB - p_GainSA) + p_GainSA;

	float Gs = (GO - p_GainSA) / (p_GainSB - p_GainSA);
	float Gss = Gs >= 0.0f && Gs <= 1.0f ? Gs * p_GainSGG + p_GainSL * (1.0f - (Gs * p_GainSGG)) : Gs;
	float GS = Gss >= 0.0f && Gss <= 1.0f ? (p_SwitchS != 1.0f ? pow(Gss, 1.0f / p_GainSG) : 1.0f - pow(1.0f - Gss, p_GainSG)) : Gss;
	float GSS = GS * (p_GainSB - p_GainSA) + p_GainSA;

	float Bs = (BO - p_GainSA) / (p_GainSB - p_GainSA);
	float Bss = Bs >= 0.0f && Bs <= 1.0f ? Bs * p_GainSGG + p_GainSL * (1.0f - (Bs * p_GainSGG)) : Bs;
	float BS = Bss >= 0.0f && Bss <= 1.0f ? (p_SwitchS != 1.0f ? pow(Bss, 1.0f / p_GainSG) : 1.0f - pow(1.0f - Bss, p_GainSG)) : Bss;
	float BSS = BS * (p_GainSB - p_GainSA) + p_GainSA;

	float Rm = (RSS - p_GainMA) / (p_GainMB - p_GainMA);
	float Rmm = Rm >= 0.0f && Rm <= 1.0f ? Rm * p_GainMGG + p_GainML * (1.0f - (Rm * p_GainMGG)) : Rm;
	float RM = Rmm >= 0.0f && Rmm <= 1.0f ? (p_SwitchM != 1.0f ? pow(Rmm, 1.0f / p_GainMG) : 1.0f - pow(1.0f - Rmm, p_GainMG)) : Rmm;
	float RMM = RM * (p_GainMB - p_GainMA) + p_GainMA;

	float Gm = (GSS - p_GainMA) / (p_GainMB - p_GainMA);
	float Gmm = Gm >= 0.0f && Gm <= 1.0f ? Gm * p_GainMGG + p_GainML * (1.0f - (Gm * p_GainMGG)) : Gm;
	float GM = Gmm >= 0.0f && Gmm <= 1.0f ? (p_SwitchM != 1.0f ? pow(Gmm, 1.0f / p_GainMG) : 1.0f - pow(1.0f - Gmm, p_GainMG)) : Gmm;
	float GMM = GM * (p_GainMB - p_GainMA) + p_GainMA;

	float Bm = (BSS - p_GainMA) / (p_GainMB - p_GainMA);
	float Bmm = Bm >= 0.0f && Bm <= 1.0f ? Bm * p_GainMGG + p_GainML * (1.0f - (Bm * p_GainMGG)) : Bm;
	float BM = Bmm >= 0.0f && Bmm <= 1.0f ? (p_SwitchM != 1.0f ? pow(Bmm, 1.0f / p_GainMG) : 1.0f - pow(1.0f - Bmm, p_GainMG)) : Bmm;
	float BMM = BM * (p_GainMB - p_GainMA) + p_GainMA;

	float Rh = (RMM - p_GainHA) / (p_GainHB - p_GainHA);
	float Rhh = Rh >= 0.0f && Rh <= 1.0f ? Rh * p_GainHGG + p_GainHL * (1.0f - (Rh * p_GainHGG)) : Rh;
	float RH = Rhh >= 0.0f && Rhh <= 1.0f ? (p_SwitchH != 1.0f ? pow(Rhh, 1.0f / p_GainHG) : 1.0f - pow(1.0f - Rhh, p_GainHG)) : Rhh;
	float RHH = RH * (p_GainHB - p_GainHA) + p_GainHA;

	float Gh = (GMM - p_GainHA) / (p_GainHB - p_GainHA);
	float Ghh = Gh >= 0.0f && Gh <= 1.0f ? Gh * p_GainHGG + p_GainHL * (1.0f - (Gh * p_GainHGG)) : Gh;
	float GH = Ghh >= 0.0f && Ghh <= 1.0f ? (p_SwitchH != 1.0f ? pow(Ghh, 1.0f / p_GainHG) : 1.0f - pow(1.0f - Ghh, p_GainHG)) : Ghh;
	float GHH = GH * (p_GainHB - p_GainHA) + p_GainHA;

	float Bh = (BMM - p_GainHA) / (p_GainHB - p_GainHA);
	float Bhh = Bh >= 0.0f && Bh <= 1.0f ? Bh * p_GainHGG + p_GainHL * (1.0f - (Bh * p_GainHGG)) : Bh;
	float BH = Bhh >= 0.0f && Bhh <= 1.0f ? (p_SwitchH != 1.0f ? pow(Bhh, 1.0f / p_GainHG) : 1.0f - pow(1.0f - Bhh, p_GainHG)) : Bhh;
	float BHH = BH * (p_GainHB - p_GainHA) + p_GainHA;
							
	p_Output[index + 0] = RHH;
	p_Output[index + 1] = GHH;
	p_Output[index + 2] = BHH;
	p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* p_Switch, float* p_Gain, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    TwelveWayAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_Switch[0], p_Switch[1], p_Switch[2], p_Switch[3], p_Gain[0], p_Gain[1], p_Gain[2],
     p_Gain[3], p_Gain[4], p_Gain[5], p_Gain[6], p_Gain[7], p_Gain[8], p_Gain[9], p_Gain[10], p_Gain[11], p_Gain[12],
      p_Gain[13], p_Gain[14], p_Gain[15], p_Gain[16], p_Gain[17], p_Input, p_Output);
}
