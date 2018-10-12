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
"__kernel void LumaKeyAdjustKernel(  \n" \
"   int p_Width,    \n" \
"   int p_Height,   \n" \
"   float p_LumaKeyR, \n" \
"   float p_LumaKeyG, \n" \
"   float p_LumaKeyB, \n" \
"   float p_LumaKeyA, \n" \
"   float p_LumaKeyD, \n" \
"   float p_LumaKeyE, \n" \
"   float p_LumaKeyO, \n" \
"   float p_LumaKeyZ, \n" \
"   float p_SwitchA, \n" \
"   float p_SwitchB, \n" \
"   __global const float* p_Input, \n" \
"   __global float* p_Output)      \n" \
"{                                  \n" \
"   float SRC[BLOCKSIZE]; \n" \
"   float w_LumaKeyR; \n" \
"   float w_LumaKeyG; \n" \
"   float w_LumaKeyB; \n" \
"   float w_LumaKeyA; \n" \
"   float w_LumaKeyD; \n" \
"   float w_LumaKeyE; \n" \
"   float w_LumaKeyO; \n" \
"   float w_LumaKeyZ; \n" \
"   float w_SwitchA; \n" \
"   float w_SwitchB; \n" \
"   float l;      \n" \
"   float L;      \n" \
"   float q;      \n" \
"   float n;      \n" \
"   float r;      \n" \
"   float g;      \n" \
"   float b;      \n" \
"   float a;      \n" \
"   float d;      \n" \
"   float e;      \n" \
"   float z;      \n" \
"   float w;      \n" \
"   float k;      \n" \
"   float alpha;  \n" \
"   float alphaM; \n" \
"   float alphaV; \n" \
"                                   \n" \
"   const int x = get_global_id(0); \n" \
"   const int y = get_global_id(1); \n" \
"                                   \n" \
"   if ((x < p_Width) && (y < p_Height)) \n" \
"   {   \n" \
"       const int index = ((y * p_Width) + x) * BLOCKSIZE; \n" \
"       \n" \
"      SRC[0] = p_Input[index + 0];     \n" \
"      SRC[1] = p_Input[index + 1];     \n" \
"      SRC[2] = p_Input[index + 2];     \n" \
"      SRC[3] = p_Input[index + 3];     \n" \
"      w_LumaKeyR  = p_LumaKeyR;   \n" \
"      w_LumaKeyG  = p_LumaKeyG;   \n" \
"      w_LumaKeyB  = p_LumaKeyB;   \n" \
"      w_LumaKeyA  = p_LumaKeyA;   \n" \
"      w_LumaKeyD  = p_LumaKeyD;   \n" \
"      w_LumaKeyE  = p_LumaKeyE;   \n" \
"      w_LumaKeyO  = p_LumaKeyO;   \n" \
"      w_LumaKeyZ  = p_LumaKeyZ;   \n" \
"      w_SwitchA = p_SwitchA;  \n" \
"      w_SwitchB = p_SwitchB;  \n" \
"                \n" \
"      l = (SRC[0] * 0.2126f) + (SRC[1] * 0.7152f) + (SRC[2] * 0.0722f);              \n" \
"      L = l - w_LumaKeyO;    \n" \
"      q = fmin(L, 1.0f);     \n" \
"      n = fmax(q, 0.0f);     \n" \
"                           \n" \
"      r = w_LumaKeyR;        \n" \
"      g = w_LumaKeyG;        \n" \
"      b = w_LumaKeyB;        \n" \
"      a = w_LumaKeyA;        \n" \
"      d = 1.0f / w_LumaKeyD;  \n" \
"      e = 1.0f / w_LumaKeyE;  \n" \
"      z = w_LumaKeyZ;        \n" \
"        \n" \
"      w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= n ? 1.0f : (r >= n ? pow((r - n) / (1.0f - g), d) : 0.0f)); \n" \
"      k = a == 1.0f ? 0.0f : (a + b <= n ? 1.0f : (a <= n ? pow((n - a) / b, e) : 0.0f)); \n" \
"      alpha = k * w;                                             \n" \
"      alphaM = alpha + (1.0f - alpha) * z;                        \n" \
"      alphaV = (w_SwitchB == 1.0f) ? 1.0f - alphaM : alphaM;      \n" \
"        \n" \
"      SRC[0] = (w_SwitchA == 1.0f) ? alphaV : SRC[0];   \n" \
"      SRC[1] = (w_SwitchA == 1.0f) ? alphaV : SRC[1];   \n" \
"      SRC[2] = (w_SwitchA == 1.0f) ? alphaV : SRC[2];     \n" \
"      SRC[3] = (w_SwitchA == 1.0f) ? SRC[3] : alphaV;  \n" \
"                                      \n" \
"      p_Output[index + 0] = SRC[0];  \n" \
"      p_Output[index + 1] = SRC[1];  \n" \
"      p_Output[index + 2] = SRC[2];  \n" \
"      p_Output[index + 3] = SRC[3];  \n" \
"   } \n" \
"}    \n" \
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

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_LumaKey, float* p_Switch, const float* p_Input, float* p_Output)
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

		kernel = clCreateKernel(program, "LumaKeyAdjustKernel", &error);
		CheckError(error, "Unable to create kernel");

		kernelMap[cmdQ] = kernel;
	}
	else
	{
		kernel = kernelMap[cmdQ];
	}

	locker.Unlock();

	int count = 0;
	error = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[2]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[3]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[4]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[5]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[6]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_LumaKey[7]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Switch[1]);
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
