#ifndef GLOBAL_H
#define GLOBAL_H


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <device_functions.h>

#include <iostream>
#include <fstream>

#define WIDTH   256
#define HEIGHT  256
#define ZOOM 1

#define VOL_WIDTH 256
#define VOL_HEIGHT 256
#define VOL_DEPTH 225


unsigned char MyTexture[HEIGHT * WIDTH * 3];
unsigned char vol[VOL_DEPTH * VOL_HEIGHT * VOL_WIDTH];

float4 colorTable[256];

texture<unsigned char, 3, cudaReadModeNormalizedFloat> texVol;
cudaArray_t cuda3DVol;

cudaArray_t cudaColorArray;
texture<float4, 1, cudaReadModeElementType> texColorTable;

//cc cuda_constant
//cd cuda_device
//cs cuda_shared
__constant__ float cc_eye[3];
__constant__ float cc_dir[3];
__constant__ float cc_cross[3];
__constant__ float cc_up[3];


//TransferFunction
int selectTransferFunction = -1;
int tableRange[2];


__device__ __host__ void normalized(float vector[3]);
__device__ __host__ float dot(float v1[3], float v2[3]);
__device__ __host__ void crossProduct(float a[3], float b[3], float c[3]);



void cudaError(cudaError_t status, char* str);
void InitCamera();
void FileRead();
void SelectTransferFunction();


void FileRead()
{
	std::ifstream myfile;
	myfile.open("bighead.den", std::ios::in | std::ios::binary);
	if (!myfile.is_open()) {
		std::cout << "file error";
	}
	myfile.read((char*)vol, VOL_DEPTH * VOL_HEIGHT * VOL_WIDTH);
	myfile.close();
}


void cudaError(cudaError_t status, char* str)
{
	if (status != cudaSuccess)
	{
		printf("%s", str);
		exit(1);
	}
}



void InitCamera()
{
	float at[3] = { 128, 128, 112 };
	float eye[3];
	float up[3] = { 0,1,0 };
	float cross[3];
	float dir[3];
	float u[3];



	printf("카메라 위치 x : ");
	scanf_s("%f", eye);
	printf("카메라 위치 y : ");
	scanf_s("%f", eye + 1);
	printf("카메라 위치 z : ");
	scanf_s("%f", eye + 2);

	dir[0] = at[0] - eye[0];
	dir[1] = at[1] - eye[1];
	dir[2] = at[2] - eye[2];


	normalized(dir);

	crossProduct(up, dir, cross);
	normalized(cross);

	crossProduct(dir, cross, u);
	normalized(u);


	printf("dir   vecotr : %f %f %f\n", dir[0], dir[1], dir[2]);
	printf("cross vecotr : %f %f %f\n", cross[0], cross[1], cross[2]);
	printf("u     vecotr : %f %f %f\n", u[0], u[1], u[2]);

	//카메라 설정을 GPU로 복사
	cudaMemcpyToSymbol(cc_eye, eye, sizeof(float) * 3);
	cudaMemcpyToSymbol(cc_dir, dir, sizeof(float) * 3);
	cudaMemcpyToSymbol(cc_cross, cross, sizeof(float) * 3);
	cudaMemcpyToSymbol(cc_up, u, sizeof(float) * 3);
}

void SelectTransferFunction()
{
	
	int selectRange[2] = { 1,2 };

	printf("Transfer Fucntion 설정\n1.BONE \n2. MUSLCE\n[%d~%d] : ", selectRange[0],selectRange[1]);
	scanf_s("%d", &selectTransferFunction);

	while (selectTransferFunction < selectRange[0] || selectTransferFunction > selectRange[1])
	{
		printf("범위초과 다시 입력하세요. \n[%d~%d] : ", selectRange[0], selectRange[1]);
		scanf_s("%d", &selectTransferFunction);
	}

	switch (selectTransferFunction)
	{
	case 1: tableRange[0] = 70; tableRange[1] = 80;  break;
	case 2: tableRange[0] = 100; tableRange[1] = 140; break;
	}
}

__device__ __host__
float dot(float v1[3], float v2[3])
{
	return (v1[0] * v2[0]) + (v1[1] * v2[1]) + (v1[2] * v2[2]);
}

__device__ __host__
void crossProduct(float a[3], float b[3], float c[3])
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __host__
void normalized(float vector[3])
{
	float size = sqrt((vector[0] * vector[0]) + (vector[1] * vector[1]) + (vector[2] * vector[2]));

	if (size == 0)
	{
		vector[0] = 0;
		vector[1] = 0;
		vector[2] = 0;
	}
	else
	{
		vector[0] /= size;
		vector[1] /= size;
		vector[2] /= size;
	}
}
#endif // !GLOBAL_H


