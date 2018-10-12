#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <stdio.h>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char *KernelSource = "\n" \
"#define  BLOCKSIZE 4 \n" \
"__kernel void TwelveWayAdjustKernel(  \n" \
"   int p_Width,                                                        \n" \
"   int p_Height,                                                       \n" \
"	float p_SwitchO,		\n" \
"	float p_SwitchS,		\n" \
"	float p_SwitchM,		\n" \
"	float p_SwitchH,		\n" \
"	float p_GainOL,		\n" \
"	float p_GainOG,		\n" \
"	float p_GainOGG,	\n" \
"	float p_GainSL,		\n" \
"	float p_GainSG,		\n" \
"	float p_GainSGG,	\n" \
"	float p_GainSA,		\n" \
"	float p_GainSB,		\n" \
"	float p_GainML,		\n" \
"	float p_GainMG,		\n" \
"	float p_GainMGG,	\n" \
"	float p_GainMA,		\n" \
"	float p_GainMB,		\n" \
"	float p_GainHL,		\n" \
"	float p_GainHG,		\n" \
"	float p_GainHGG,	\n" \
"	float p_GainHA,		\n" \
"	float p_GainHB,		\n" \
"   __global const float* p_Input,                                      \n" \
"   __global float* p_Output)                                           \n" \
"{                                                                      \n" \
"   float SRC[BLOCKSIZE]; \n" \
"	float w_SwitchO;		\n" \
"	float w_SwitchS;		\n" \
"	float w_SwitchM;		\n" \
"	float w_SwitchH;		\n" \
"	float w_GainOL;		\n" \
"	float w_GainOG;		\n" \
"	float w_GainOGG;	\n" \
"	float w_GainSL;		\n" \
"	float w_GainSG;		\n" \
"	float w_GainSGG;	\n" \
"	float w_GainSA;		\n" \
"	float w_GainSB;		\n" \
"	float w_GainML;		\n" \
"	float w_GainMG;		\n" \
"	float w_GainMGG;	\n" \
"	float w_GainMA;		\n" \
"	float w_GainMB;		\n" \
"	float w_GainHL;		\n" \
"	float w_GainHG;		\n" \
"	float w_GainHGG;	\n" \
"	float w_GainHA;		\n" \
"	float w_GainHB;		\n" \
"    float Ro;       \n" \
"    float RO;       \n" \
"    float Go;       \n" \
"    float GO;       \n" \
"    float Bo;       \n" \
"    float BO;       \n" \
"    float Rs;       \n" \
"    float Rss;       \n" \
"    float RS;       \n" \
"    float RSS;       \n" \
"    float Gs;       \n" \
"    float Gss;       \n" \
"    float GS;       \n" \
"    float GSS;       \n" \
"    float Bs;       \n" \
"    float Bss;       \n" \
"    float BS;       \n" \
"    float BSS;       \n" \
"    float Rm;       \n" \
"    float Rmm;       \n" \
"    float RM;       \n" \
"    float RMM;       \n" \
"    float Gm;       \n" \
"    float Gmm;       \n" \
"    float GM;       \n" \
"    float GMM;       \n" \
"    float Bm;       \n" \
"    float Bmm;       \n" \
"    float BM;       \n" \
"    float BMM;       \n" \
"    float Rh;       \n" \
"    float Rhh;       \n" \
"    float RH;       \n" \
"    float RHH;       \n" \
"    float Gh;       \n" \
"    float Ghh;       \n" \
"    float GH;       \n" \
"    float GHH;       \n" \
"    float Bh;       \n" \
"    float Bhh;       \n" \
"    float BH;       \n" \
"    float BHH;       \n" \
"   const int x = get_global_id(0);                                     \n" \
"   const int y = get_global_id(1);                                     \n" \
"                                                                       \n" \
"   if ((x < p_Width) && (y < p_Height))                                \n" \
"   {                                                                   \n" \
"      const int index = ((y * p_Width) + x) * BLOCKSIZE;               \n" \
"                                                                       \n" \
"      SRC[0] = p_Input[index + 0] ;    \n" \
"      SRC[1] = p_Input[index + 1] ;    \n" \
"      SRC[2] = p_Input[index + 2] ;    \n" \
"      SRC[3] = p_Input[index + 3] ;    \n" \
"      w_SwitchO   = p_SwitchO;  \n" \
"      w_SwitchS   = p_SwitchS; \n" \
"      w_SwitchM   = p_SwitchM; \n" \
"      w_SwitchH   = p_SwitchH; \n" \
"      w_GainOL  = p_GainOL;    \n" \
"      w_GainOG  = p_GainOG;    \n" \
"      w_GainOGG   = p_GainOGG; \n" \
"      w_GainSL  = p_GainSL;    \n" \
"      w_GainSG  = p_GainSG;    \n" \
"      w_GainSGG   = p_GainSGG; \n" \
"      w_GainSA  = p_GainSA;    \n" \
"      w_GainSB  = p_GainSB;    \n" \
"      w_GainML  = p_GainML;    \n" \
"      w_GainMG  = p_GainMG;    \n" \
"      w_GainMGG   = p_GainMGG; \n" \
"      w_GainMA  = p_GainMA;    \n" \
"      w_GainMB  = p_GainMB;    \n" \
"      w_GainHL  = p_GainHL;    \n" \
"      w_GainHG  = p_GainHG;    \n" \
"      w_GainHGG   = p_GainHGG; \n" \
"      w_GainHA  = p_GainHA;    \n" \
"      w_GainHB  = p_GainHB;    \n" \
"                  \n" \
"    Ro = SRC[0] * w_GainOGG + w_GainOL * (1.0f - (SRC[0] * w_GainOGG));	\n" \
"    RO = Ro >= 0.0f && Ro <= 1.0f ? (w_SwitchO != 1.0f ? pow(Ro, 1.0f / w_GainOG) : 1.0f - pow(1.0f - Ro, w_GainOG)) : Ro;	\n" \
"    				\n" \
"    Go = SRC[1] * w_GainOGG + w_GainOL * (1.0f - (SRC[1] * w_GainOGG));	\n" \
"    GO = Go >= 0.0f && Go <= 1.0f ? (w_SwitchO != 1.0f ? pow(Go, 1.0f / w_GainOG) : 1.0f - pow(1.0f - Go, w_GainOG)) : Go;	\n" \
"    				\n" \
"    Bo = SRC[2] * w_GainOGG + w_GainOL * (1.0f - (SRC[2] * w_GainOGG));	\n" \
"    BO = Bo >= 0.0f && Bo <= 1.0f ? (w_SwitchO != 1.0f ? pow(Bo, 1.0f / w_GainOG) : 1.0f - pow(1.0f - Bo, w_GainOG)) : Bo;	\n" \
"					\n" \
"	 Rs = (RO - w_GainSA) / (w_GainSB - w_GainSA);	\n" \
"	 Rss = Rs >= 0.0f && Rs <= 1.0f ? Rs * w_GainSGG + w_GainSL * (1.0f - (Rs * w_GainSGG)) : Rs;	\n" \
"	 RS = Rss >= 0.0f && Rss <= 1.0f ? (w_SwitchS != 1.0f ? pow(Rss, 1.0f / w_GainSG) : 1.0f - pow(1.0f - Rss, w_GainSG)) : Rss;	\n" \
"	 RSS = RS * (w_GainSB - w_GainSA) + w_GainSA;	\n" \
"				\n" \
"	 Gs = (GO - w_GainSA) / (w_GainSB - w_GainSA);	\n" \
"	 Gss = Gs >= 0.0f && Gs <= 1.0f ? Gs * w_GainSGG + w_GainSL * (1.0f - (Gs * w_GainSGG)) : Gs;	\n" \
"	 GS = Gss >= 0.0f && Gss <= 1.0f ? (w_SwitchS != 1.0f ? pow(Gss, 1.0f / w_GainSG) : 1.0f - pow(1.0f - Gss, w_GainSG)) : Gss;	\n" \
"	 GSS = GS * (w_GainSB - w_GainSA) + w_GainSA;	\n" \
"				\n" \
"	 Bs = (BO - w_GainSA) / (w_GainSB - w_GainSA);	\n" \
"	 Bss = Bs >= 0.0f && Bs <= 1.0f ? Bs * w_GainSGG + w_GainSL * (1.0f - (Bs * w_GainSGG)) : Bs;	\n" \
"	 BS = Bss >= 0.0f && Bss <= 1.0f ? (w_SwitchS != 1.0f ? pow(Bss, 1.0f / w_GainSG) : 1.0f - pow(1.0f - Bss, w_GainSG)) : Bss;	\n" \
"	 BSS = BS * (w_GainSB - w_GainSA) + w_GainSA;	\n" \
"				\n" \
"	 Rm = (RSS - w_GainMA) / (w_GainMB - w_GainMA);	\n" \
"	 Rmm = Rm >= 0.0f && Rm <= 1.0f ? Rm * w_GainMGG + w_GainML * (1.0f - (Rm * w_GainMGG)) : Rm;	\n" \
"	 RM = Rmm >= 0.0f && Rmm <= 1.0f ? (w_SwitchM != 1.0f ? pow(Rmm, 1.0f / w_GainMG) : 1.0f - pow(1.0f - Rmm, w_GainMG)) : Rmm;	\n" \
"	 RMM = RM * (w_GainMB - w_GainMA) + w_GainMA;	\n" \
"				\n" \
"	 Gm = (GSS - w_GainMA) / (w_GainMB - w_GainMA);	\n" \
"	 Gmm = Gm >= 0.0f && Gm <= 1.0f ? Gm * w_GainMGG + w_GainML * (1.0f - (Gm * w_GainMGG)) : Gm;	\n" \
"	 GM = Gmm >= 0.0f && Gmm <= 1.0f ? (w_SwitchM != 1.0f ? pow(Gmm, 1.0f / w_GainMG) : 1.0f - pow(1.0f - Gmm, w_GainMG)) : Gmm;	\n" \
"	 GMM = GM * (w_GainMB - w_GainMA) + w_GainMA;	\n" \
"				\n" \
"	 Bm = (BSS - w_GainMA) / (w_GainMB - w_GainMA);	\n" \
"	 Bmm = Bm >= 0.0f && Bm <= 1.0f ? Bm * w_GainMGG + w_GainML * (1.0f - (Bm * w_GainMGG)) : Bm;	\n" \
"	 BM = Bmm >= 0.0f && Bmm <= 1.0f ? (w_SwitchM != 1.0f ? pow(Bmm, 1.0f / w_GainMG) : 1.0f - pow(1.0f - Bmm, w_GainMG)) : Bmm;	\n" \
"	 BMM = BM * (w_GainMB - w_GainMA) + w_GainMA;	\n" \
"				\n" \
"	 Rh = (RMM - w_GainHA) / (w_GainHB - w_GainHA);	\n" \
"	 Rhh = Rh >= 0.0f && Rh <= 1.0f ? Rh * w_GainHGG + w_GainHL * (1.0f - (Rh * w_GainHGG)) : Rh;	\n" \
"	 RH = Rhh >= 0.0f && Rhh <= 1.0f ? (w_SwitchH != 1.0f ? pow(Rhh, 1.0f / w_GainHG) : 1.0f - pow(1.0f - Rhh, w_GainHG)) : Rhh;	\n" \
"	 RHH = RH * (w_GainHB - w_GainHA) + w_GainHA;	\n" \
"				\n" \
"	 Gh = (GMM - w_GainHA) / (w_GainHB - w_GainHA);	\n" \
"	 Ghh = Gh >= 0.0f && Gh <= 1.0f ? Gh * w_GainHGG + w_GainHL * (1.0f - (Gh * w_GainHGG)) : Gh;	\n" \
"	 GH = Ghh >= 0.0f && Ghh <= 1.0f ? (w_SwitchH != 1.0f ? pow(Ghh, 1.0f / w_GainHG) : 1.0f - pow(1.0f - Ghh, w_GainHG)) : Ghh;	\n" \
"	 GHH = GH * (w_GainHB - w_GainHA) + w_GainHA;	\n" \
"				\n" \
"	 Bh = (BMM - w_GainHA) / (w_GainHB - w_GainHA);	\n" \
"	 Bhh = Bh >= 0.0f && Bh <= 1.0f ? Bh * w_GainHGG + w_GainHL * (1.0f - (Bh * w_GainHGG)) : Bh;	\n" \
"	 BH = Bhh >= 0.0f && Bhh <= 1.0f ? (w_SwitchH != 1.0f ? pow(Bhh, 1.0f / w_GainHG) : 1.0f - pow(1.0f - Bhh, w_GainHG)) : Bhh;	\n" \
"	 BHH = BH * (w_GainHB - w_GainHA) + w_GainHA;	\n" \
"														\n" \
"       SRC[0] = RHH;        \n" \
"       SRC[1] = GHH;        \n" \
"       SRC[2] = BHH;        \n" \
"                                      \n" \
"       p_Output[index + 0] = SRC[0];  \n" \
"       p_Output[index + 1] = SRC[1];  \n" \
"       p_Output[index + 2] = SRC[2];  \n" \
"       p_Output[index + 3] = SRC[3];  \n" \
"                                      \n" \
"   }                                  \n" \
"}                                     \n" \
"\n";

class Locker
{
public:
	Locker()
	{
#ifdef _WIN64
		InitializeCriticalSection(&mutex);
#else
		pthread_mutex_init(&mutex, NULL);
#endif
	}

	~Locker()
	{
#ifdef _WIN64
		DeleteCriticalSection(&mutex);
#else
		pthread_mutex_destroy(&mutex);
#endif
	}

	void Lock()
	{
#ifdef _WIN64
		EnterCriticalSection(&mutex);
#else
		pthread_mutex_lock(&mutex);
#endif
	}

	void Unlock()
	{
#ifdef _WIN64
		LeaveCriticalSection(&mutex);
#else
		pthread_mutex_unlock(&mutex);
#endif
	}

private:
#ifdef _WIN64
	CRITICAL_SECTION mutex;
#else
	pthread_mutex_t mutex;
#endif
};


void CheckError(cl_int p_Error, const char* p_Msg)
{
	if (p_Error != CL_SUCCESS)
	{
		fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
	}
}

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Switch, float* p_Gain, const float* p_Input, float* p_Output)
{
	cl_int error;

	cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

	// store device id and kernel per command queue (required for multi-GPU systems)
	static std::map<cl_command_queue, cl_device_id> deviceIdMap;
	static std::map<cl_command_queue, cl_kernel> kernelMap;

	static Locker locker; // simple lock to control access to the above maps from multiple threads

	locker.Lock();

	// find the device id corresponding to the command queue
	cl_device_id deviceId = NULL;
	if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
	{
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
		CheckError(error, "Unable to get the device");

		deviceIdMap[cmdQ] = deviceId;
	}
	else
	{
		deviceId = deviceIdMap[cmdQ];
	}

//#define _DEBUG


	// find the program kernel corresponding to the command queue
	cl_kernel kernel = NULL;
	if (kernelMap.find(cmdQ) == kernelMap.end())
	{
		cl_context clContext = NULL;
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
		CheckError(error, "Unable to get the context");

		cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
		CheckError(error, "Unable to create program");

		error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#ifdef _DEBUG
		if (error != CL_SUCCESS)
		{
			char buffer[4096];
			size_t length;
			clGetProgramBuildInfo
				(
				program,
				// valid program object
				deviceId,
				// valid device_id that executable was built
				CL_PROGRAM_BUILD_LOG,
				// indicate to retrieve build log
				sizeof(buffer),
				// size of the buffer to write log to
				buffer,
				// the actual buffer to write log to
				&length);
			// the actual size in bytes of data copied to buffer
			FILE * pFile;
			pFile = fopen("/", "w");
			if (pFile != NULL)
			{
				fprintf(pFile, "%s\n", buffer);
				//fprintf(pFile, "%s [%lu]\n", "localWorkSize 0 =", szWorkSize);
			}
			fclose(pFile);
		}
#else
		CheckError(error, "Unable to build program");
#endif

		kernel = clCreateKernel(program, "TwelveWayAdjustKernel", &error);
		CheckError(error, "Unable to create kernel");

		kernelMap[cmdQ] = kernel;
	}
	else
	{
		kernel = kernelMap[cmdQ];
	}

	locker.Unlock();

    int count = 0;
    error  = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[3]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[3]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[4]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[5]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[6]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[7]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[8]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[9]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[10]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[11]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[12]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[13]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[14]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[15]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[16]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[17]);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
