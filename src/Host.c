#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
//#else
//#include<sys/time.h>
#endif

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
//#include<conio.h>//included for _getch() only
#include<CL/cl.h>
#include"cl_errors.h"

//Mersenne Twister parameter directory
//#define MT_DATA_FILE "../mtParameters/mtDATA/mtDATA_521_32_30.bin"
#define MT_DATA_FILE "param.bin"

// ** OpenCL PARAMETERS ** //
#define MAX_PLATFORM_NUMBER 4
#define MAX_DEVICE_NUMBER 8
#define MAX_DEVICE_NUMBER_PER_PLATFORM 4

// ** CONTROL PARAMETERS ** //
#define D 2 // dimension of problem
#define NP 2048 // size of population
#define F 0.9 // differentiation constant
#define CR 0.5 // crossover constant
#define GEN 1200 // number of generations
#define L -10.0 // low boundary constraint
#define H 10.0 // high boundary constraint

/* same as mt_struct without the work vector */
typedef struct
{
	cl_uint aaa;
	cl_int mm, nn, rr, ww;
	cl_uint wmask, umask, lmask;
	cl_int shift0, shift1, shiftB, shiftC;
	cl_uint maskB, maskC;
	cl_int i;
}mt_struct_naked;

typedef struct
{
	cl_platform_id platform[MAX_PLATFORM_NUMBER];
	cl_device_id selectedDevice[MAX_DEVICE_NUMBER]; /* compute device id */
	cl_context context; /* compute context */
	cl_command_queue commandQueue; /* compute command queue */
	cl_program program;
	cl_uint num_platforms, selectedDeviceIndex;
}dev_info;

// ** Algorithm Variables ** //
typedef struct
{
	double X[D]; // trial vector
	double Pop[D][NP]; // population
	double Fit[NP]; // fitness of the population
	double f[NP]; // fitness of the trial individual
	int iBest; // index of the best solution
	int Rnd; // mutation parameter
}algorithmVariables;

//clGetDeviceInfo(di->device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);

char* convertToString(const char* filename)
{
	char *file_contents;
	size_t kernel_size;
	FILE *input_file = fopen(filename, "rb");
	if (!input_file)
	{
		perror("ERROR CONVERTING SOURCE TO STRING ");
		exit(EXIT_FAILURE);
	}//end if
	fseek(input_file, 0, SEEK_END);		//find and go to the end of the file
	kernel_size = ftell(input_file);	//end of the file = string size
	rewind(input_file);					//return to the beginning of the file
	file_contents = (char*)malloc(kernel_size + 1);//+1 for NULL char
	file_contents[kernel_size] = '\0';	//add NULL char at the end of the string
	fread(file_contents, sizeof(char), kernel_size, input_file);//read file to file_contents
	fclose(input_file);
	return file_contents;
}//end convertToString

void printOCLDevices(cl_platform_id platform[], cl_device_id selectedDevice[], const cl_uint numPlatforms)
{
	cl_device_id platform_device[MAX_DEVICE_NUMBER_PER_PLATFORM], all_device[MAX_DEVICE_NUMBER];
	cl_uint num_platform_devices, all_device_index = 0;
	cl_ulong global_memory, max_alocatable_memory;
	cl_uint num_cu;
	cl_int error;
	char platform_vendor[120];
	char device_name[120];
	for (cl_uint counter = 0; counter < numPlatforms; counter++)//platform loop
	{
		error = clGetPlatformInfo(platform[counter], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL); clChkErr(error);

		for (cl_uint counter2 = printf("OpenCL Platform #%u: %s\n", counter + 1, platform_vendor); counter2 > 0; counter2--)//print platform
		{
			putchar('=');
		}//end for
		putchar('\n');

		error = clGetDeviceIDs(platform[counter], CL_DEVICE_TYPE_ALL, MAX_DEVICE_NUMBER_PER_PLATFORM, platform_device, &num_platform_devices); clChkErr(error);//get devices

		for (cl_uint counter2 = 0; counter2 < num_platform_devices; counter2++)//ordering devices independently from platform
		{
			all_device[all_device_index++] = platform_device[counter2];
		}//end for

		for (cl_uint counter2 = all_device_index - num_platform_devices; counter2 < all_device_index; counter2++)//device loop
		{
			char* is_selected = "Skipped";
			clGetDeviceInfo(all_device[counter2], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
			clGetDeviceInfo(all_device[counter2], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alocatable_memory), &max_alocatable_memory, NULL);
			clGetDeviceInfo(all_device[counter2], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_memory), &global_memory, NULL);
			clGetDeviceInfo(all_device[counter2], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_cu), &num_cu, NULL);
			for (cl_uint counter3 = 0; counter3 < all_device_index; counter3++)
			{
				if (all_device[counter2] == selectedDevice[counter3])
				{
					is_selected = "Using";
				}//end if
			}//end for
			printf("%s Device #%u: %s, %0.0f/%0.0f MB allocatable, %u Compute Units\n", is_selected, counter2 + 1, device_name, (float)max_alocatable_memory / 1024 / 1024, (float)global_memory / 1024 / 1024, num_cu);
		}//end for
		putchar('\n');
	}//end for
}//end printOCLDevices

void readMTdata(mt_struct_naked *mts, const int Nmts)
{
	FILE *fp;
	int fileN, i;
//	mt_struct_naked lmts;

	fp = fopen(MT_DATA_FILE, "rb");
	if (NULL == fp)
	{
		perror("MT data file not found ");
		exit(EXIT_FAILURE);
	}//end if

	/*Read number of mt_struct_naked stored in file*/
	fread(&fileN, sizeof(int), 1, fp);

	/*If not enough data, exit*/
	if (fileN < Nmts)
	{
		puts("Number of data in file is less than # of threds. Run mtParams again.");
		exit(EXIT_FAILURE);
	}//end if

	/*Read Nmts structs from file*/
	for (i = 0; i < Nmts; i++)
	{
		fread(mts + i, sizeof(mt_struct_naked), 1, fp);
	}//end for

	fclose(fp);
}

void createPopulation(dev_info* di, mt_struct_naked* mts, cl_mem* pop, cl_mem* trial, cl_mem* fitness, cl_mem* individual)
{
	int *seeds;
	unsigned int *states;
	cl_int error;

	size_t initLocalSize = (size_t)D;//to be determined
	size_t localSize = (size_t)D;
	size_t globalSize = (size_t)D * NP;

	seeds = (int*)malloc(sizeof(int) * globalSize);

	srand((unsigned int)time(NULL));
	for (size_t i = 0; i < globalSize; i++)
	{
		seeds[i] = rand();
	}//end for

	/* Setting arrays states*/
	states = (unsigned int *)malloc(sizeof(unsigned int) * mts[0].nn * globalSize);
	if (states == NULL)
	{
		printf("Out of memory!\n");
		exit(EXIT_FAILURE);
	}//end if

	printf("Total # of samples = %d\n", D * NP);

	/* Create buffer objects */
	cl_mem mtsBuffer = clCreateBuffer(di->context, CL_MEM_READ_ONLY, sizeof(mt_struct_naked) * globalSize, NULL, &error); clChkErr(error);
	cl_mem seedBuffer = clCreateBuffer(di->context, CL_MEM_READ_ONLY, sizeof(int) * globalSize, NULL, &error); clChkErr(error);
	cl_mem stateBuffer = clCreateBuffer(di->context, CL_MEM_READ_WRITE, sizeof(unsigned int) * mts[0].nn * globalSize, NULL, &error); clChkErr(error);

	/* Enqueue buffers to the selected device */
	error = clEnqueueWriteBuffer(di->commandQueue, mtsBuffer, CL_NON_BLOCKING, 0, sizeof(mt_struct_naked) * globalSize, mts, 0, NULL, NULL); clChkErr(error);
	error = clEnqueueWriteBuffer(di->commandQueue, seedBuffer, CL_NON_BLOCKING, 0, sizeof(int) * globalSize, seeds, 0, NULL, NULL); clChkErr(error);
	error = clEnqueueWriteBuffer(di->commandQueue, stateBuffer, CL_NON_BLOCKING, 0, sizeof(unsigned int) * mts[0].nn * globalSize, states, 0, NULL, NULL); clChkErr(error);
	clFinish(di->commandQueue);//block until written

	/*Set initRNG kernel and its arguments */
	cl_kernel kernel = clCreateKernel(di->program, "initRNG", &error); clChkErr(error);

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mtsBuffer); clChkErr(error);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &stateBuffer); clChkErr(error);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &seedBuffer); clChkErr(error);

	error = clEnqueueNDRangeKernel(di->commandQueue, kernel, 1, NULL, &globalSize, &initLocalSize, 0, NULL, NULL); clChkErr(error);
	error = clFinish(di->commandQueue); clChkErr(error);

	clReleaseMemObject(seedBuffer);
	clReleaseKernel(kernel);

	/*Set createPopulation kernel and its arguments*/
	kernel = clCreateKernel(di->program, "createPopulationHolderTable", &error); clChkErr(error);

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mtsBuffer); clChkErr(error);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &stateBuffer); clChkErr(error);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), pop); clChkErr(error);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), trial); clChkErr(error);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), fitness); clChkErr(error);

	error = clEnqueueNDRangeKernel(di->commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL); clChkErr(error);
	error = clFinish(di->commandQueue); clChkErr(error);

	clReleaseKernel(kernel);

	/*Set optimization kernel and its arguments*/
	kernel = clCreateKernel(di->program, "optimizationHolderTable", &error); clChkErr(error);

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mtsBuffer); clChkErr(error);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &stateBuffer); clChkErr(error);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), pop); clChkErr(error);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), trial); clChkErr(error);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), fitness); clChkErr(error);
	error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), individual); clChkErr(error);
	//error |= clSetKernelArg(kernel, 6, sizeof(cl_mem), bestIndex); clChkErr(error);

	error = clEnqueueNDRangeKernel(di->commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL); clChkErr(error);
	error = clFinish(di->commandQueue); clChkErr(error);

	/* free memory objects */
	clReleaseMemObject(mtsBuffer);
	clReleaseMemObject(stateBuffer);
	clReleaseKernel(kernel);
	free(seeds);
	free(states);
}//end createPopulation

int main(int argc, char* argv[])
{
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	algorithmVariables algorithmVariable;

	// ** OpenCL Variables ** //
//	size_t maxWI;
	dev_info di;
	cl_int error;

	const char *filename = "DEkernels.cl";//program source filename

	/* Set initial data */
	di.selectedDeviceIndex = 1;//we want 1 device in a platform
	size_t globalSize;//, localSize = 1;//to be determined
	globalSize = D * NP;

	/* Read PRNG data from file and set the initial seeds for the threads */
	mt_struct_naked *mts = (mt_struct_naked*)malloc(globalSize * sizeof(mt_struct_naked));
	readMTdata(mts, (int)globalSize);

	//puts("Starting...\n");

//	clGetDeviceInfo(di.selectedDevice[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWI, NULL);//get selected device's max wg

	/* Set compute device */
	error = clGetPlatformIDs(MAX_PLATFORM_NUMBER, di.platform, &di.num_platforms); clChkErr(error);//get platforms

	error = clGetDeviceIDs(di.platform[0], CL_DEVICE_TYPE_DEFAULT, MAX_DEVICE_NUMBER_PER_PLATFORM, di.selectedDevice, &(di.selectedDeviceIndex)); clChkErr(error);//get devices in the first platform

	di.context = clCreateContext(NULL, di.selectedDeviceIndex, di.selectedDevice, NULL, NULL, &error); clChkErr(error);//context with only one device in it

	di.commandQueue = clCreateCommandQueue(di.context, di.selectedDevice[0], 0, &error); clChkErr(error);//single command queue for the first (single) device

//	printOCLDevices(di.platform, di.selectedDevice, di.num_platforms);// Printing Devices
//	system("@cls||clear");//clear OpenCL Info or the Screen

	char* source = convertToString(filename);//get program source

//	puts(source);//print program source(kernels) for debugging purposes

	di.program = clCreateProgramWithSource(di.context, 1, (const char**)&source, NULL, &error); clChkErr(error);//build program

	/*Pass array sizes as defines using -D flag */
	char buildOptions[120];
	sprintf(buildOptions, "-D MT_NN=%d -D H=%f -D L=%f -D GEN=%d -D CR=%f -D F=%f -D D=%d -D NP=%d", mts[0].nn, H, L, GEN, CR, F, D, NP);

	if (clBuildProgram(di.program, 0, NULL, buildOptions, NULL, NULL) != CL_SUCCESS)//print the result of build log in case of build errors
	{
		char log[10240];
		clGetProgramBuildInfo(di.program, di.selectedDevice[0], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
		puts("Error Building Program\nBuild Log:");
		puts(log);
		exit(EXIT_FAILURE);
	}//end if

	cl_mem population = clCreateBuffer(di.context, CL_MEM_WRITE_ONLY, sizeof(double) * D * NP, NULL, &error); clChkErr(error);
	cl_mem trialVector = clCreateBuffer(di.context, CL_MEM_WRITE_ONLY, sizeof(double) * D, NULL, &error); clChkErr(error);
	cl_mem populationFitness = clCreateBuffer(di.context, CL_MEM_WRITE_ONLY, sizeof(double) * NP, NULL, &error); clChkErr(error);
	cl_mem individualFitness = clCreateBuffer(di.context, CL_MEM_WRITE_ONLY, sizeof(double) * NP, NULL, &error); clChkErr(error);
	//cl_mem bestIndex = clCreateBuffer(di.context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &error); clChkErr(error);

	//Creation Of Population
	createPopulation(&di, mts, &population, &trialVector, &populationFitness, &individualFitness);// initialize each individual within boundary constraints

	//Read variables
	error = clEnqueueReadBuffer(di.commandQueue, population, CL_NON_BLOCKING, 0, sizeof(double) * D * NP, algorithmVariable.Pop, 0, NULL, NULL); clChkErr(error);
	error = clEnqueueReadBuffer(di.commandQueue, trialVector, CL_NON_BLOCKING, 0, sizeof(double) * D, algorithmVariable.X, 0, NULL, NULL); clChkErr(error);
	error = clEnqueueReadBuffer(di.commandQueue, populationFitness, CL_NON_BLOCKING, 0, sizeof(double) * NP, algorithmVariable.Fit, 0, NULL, NULL); clChkErr(error);
	error = clEnqueueReadBuffer(di.commandQueue, individualFitness, CL_NON_BLOCKING, 0, sizeof(double) * NP, algorithmVariable.f, 0, NULL, NULL); clChkErr(error);
	//error = clEnqueueReadBuffer(di.commandQueue, bestIndex, CL_NON_BLOCKING, 0, sizeof(int), &algorithmVariable.iBest, 0, NULL, NULL); clChkErr(error);
	clFinish(di.commandQueue);//block

	algorithmVariable.iBest = 0;
	for (int j = 1; j < NP; j++)//determine best index(find minimum)
	{
		// if trial is better than the best
		if (algorithmVariable.Fit[j] <= algorithmVariable.Fit[algorithmVariable.iBest])
		{
			algorithmVariable.iBest = j;//update the best index(minimum index)
		}//end if
	}//end for

	gettimeofday(&tv2, NULL);

	/*
	//print to stdout
	puts("OPTIMUM : ");
	for (int i = 0; i < D; i++)
	{
		for (algorithmVariable.iBest = 0; algorithmVariable.iBest < NP; algorithmVariable.iBest++)
		{
			printf("%g\n", algorithmVariable.Pop[i][algorithmVariable.iBest]);
		}//end for
	}//end for
	*/
	/*
	//print to stdout
	puts("Population : ");
	for (int i = 0; i < D; i++)
	{
		for(iBest = 0; iBest < NP; iBest++)
		printf("Pop[%d][%d]= %g\n", i, iBest, Pop[i][iBest]);
	}//end for

	puts("Trial : ");
	for (int i = 0; i < D; i++)
	{
		printf("X[%d] = %f\n", i, X[i]);
	}//end for

	puts("Fitness : ");
	for (int i = 0; i < NP; i++)
	{
		printf("Fit[%d] = %f\n", i, Fit[i]);
	}//end for
	*/

	//print to a txt
	FILE* fp;
	const char* foutput = "RESULT.txt";
	fp = fopen(foutput, "w");

	fprintf(fp, "Execution Time = %f seconds\n", (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec));
	fprintf(fp, "Best Index is = %d\n", algorithmVariable.iBest);
	fprintf(fp, "BEST FITNESS = %g\n", algorithmVariable.Fit[algorithmVariable.iBest]);
	fprintf(fp, "GEN = %d", GEN);

	fputs("Population Fitness : \n", fp);
	for (int i = 0; i < NP; i++)
	{
		fprintf(fp, "Fit[%d] = %f\n", i, algorithmVariable.Fit[i]);
	}//end for

	fputs("Individual Fitness : \n", fp);
	for (int i = 0; i < NP; i++)
	{
		fprintf(fp, "f[%d] = %f\n", i, algorithmVariable.f[i]);
	}//end for

	fputs("Trial : \n", fp);
	for (int i = 0; i < D; i++)
	{
		fprintf(fp, "X[%d] = %f\n", i, algorithmVariable.X[i]);
	}//end for

	fputs("Population : \n", fp);
	for (int i = 0; i < D; i++)
	{
		for (int i2 = 0; i2 < NP; i2++)
		{
			fprintf(fp, "Pop[%d][%d]= %g\n", i, i2, algorithmVariable.Pop[i][i2]);
		}//end for
	}//end for

	fclose(fp);
	printf("Check \"%s\" for output\n", foutput);

	//cleanup
	clReleaseMemObject(population);
	clReleaseMemObject(trialVector);
	clReleaseMemObject(populationFitness);
	clReleaseMemObject(individualFitness);
	//clReleaseMemObject(bestIndex);
	clReleaseProgram(di.program);
	clReleaseCommandQueue(di.commandQueue);
	clReleaseContext(di.context);
	free(mts);
	if (source != NULL)
	{
		free(source);
		source = NULL;
	}//end if
	//_getch();//press any key to continue (unportable)
	return 0;
}//end main
