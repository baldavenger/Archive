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
"__kernel void LiftGammaGainAdjustKernel(\n" \
"   int p_Width,\n" \
"   int p_Height,\n" \
"   float p_GainL,\n" \
"   float p_GainG,\n" \
"   float p_GainGG,\n" \
"   float p_GainO,\n" \
"   __global const float* p_Input,\n" \
"   __global float* p_Output)\n" \
"{\n" \
"   float SRC[BLOCKSIZE]; \n" \
"   float w_GainL;\n" \
"   float w_GainG;\n" \
"   float w_GainGG;\n" \
"   float w_GainO;\n" \
"   const int x = get_global_id(0);\n" \
"   const int y = get_global_id(1);\n" \
"\n" \
"   if ((x < p_Width) && (y < p_Height))\n" \
"   {\n" \
"      const int index = ((y * p_Width) + x) * BLOCKSIZE;   \n" \
"      SRC[0] = p_Input[index + 0];    \n" \
"      SRC[1] = p_Input[index + 1];    \n" \
"      SRC[2] = p_Input[index + 2];    \n" \
"      SRC[3] = p_Input[index + 3];    \n" \
"	   w_GainL = p_GainL;\n" \
"	   w_GainG = p_GainG;\n" \
"	   w_GainGG = p_GainGG;\n" \
"	   w_GainO = p_GainO;\n" \
"\n" \
"	 p_Output[index + 0] = pow((SRC[0] * w_GainGG) + (w_GainL * (1.0 - (SRC[0] * w_GainGG))) + w_GainO, 1.0/w_GainG);\n" \
"    p_Output[index + 1] = pow((SRC[1] * w_GainGG) + (w_GainL * (1.0 - (SRC[1] * w_GainGG))) + w_GainO, 1.0/w_GainG);\n" \
"    p_Output[index + 2] = pow((SRC[2] * w_GainGG) + (w_GainL * (1.0 - (SRC[2] * w_GainGG))) + w_GainO, 1.0/w_GainG);\n" \
"    p_Output[index + 3] = SRC[3];\n" \
"   }\n" \
"}\n" \
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
void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output)
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

		kernel = clCreateKernel(program, "LiftGammaGainAdjustKernel", &error);
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
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Gain[3]);
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
