#ifndef TRASNFERF_UNCTION_H
#define TRASNFERF_UNCTION_H

#include "Global.h"

//TransferFunction색상 지정
float aColor[3] = { 0.6, 0.6, 0.6 };
float bColor[3] = { 1.0, 1.0, 1.0 };


//color테이블 생성
void makeColorTable(int a, int b, float acol[3], float bcol[3]);
//alpha테이블 생성
void makeAlphaTable(int a, int b);
//TransferFunction 설정
void SetTransferFunction();

void makeColorTable(int a, int b, float acol[3], float bcol[3])
{
	for (int i = 0; i < a; ++i)
	{
		colorTable[i].x = (float)((i * acol[0]) / (a));
		colorTable[i].y = (float)((i * acol[1]) / (a));
		colorTable[i].z = (float)((i * acol[2]) / (a));
	}
	// a 대입시 0 
	// b-1 대입시 (b - 1 - a) / (b-a) = 1
	for (int i = a; i < b; ++i)
	{
		colorTable[i].x = (float)((b - i) * acol[0] + (i - a) * bcol[2]) / (b - a);
		colorTable[i].y = (float)((b - i) * acol[1] + (i - a) * bcol[2]) / (b - a);
		colorTable[i].z = (float)((b - i) * acol[2] + (i - a) * bcol[2]) / (b - a);
	}
	for (int i = b; i < 256; ++i)
	{
		colorTable[i].x = (float)((255 - i) * bcol[0] + (i - b)) / (255 - b);
		colorTable[i].y = (float)((255 - i) * bcol[1] + (i - b)) / (255 - b);
		colorTable[i].z = (float)((255 - i) * bcol[2] + (i - b)) / (255 - b);

	}

}


void makeAlphaTable(int a, int b)
{
	for (int i = 0; i < a; ++i)
	{
		colorTable[i].w = 0;
	}
	// a 대입시 0 
	// b-1 대입시 (b - 1 - a) / (b-a) = 1
	for (int i = a; i < b; ++i)
	{
		colorTable[i].w = (float)(i - a) / (b - a);
	}
	for (int i = b; i < 256; ++i)
	{
		colorTable[i].w = 1;
	}
}


void SetTransferFunction() 
{



	//100~140까지만 보이게 (뼈만 보이게)
	makeColorTable(tableRange[0], tableRange[1], aColor, bColor);
	makeAlphaTable(tableRange[0], tableRange[1]);

	cudaError_t status;

	//CUDA Texture를 만들기 위한 Desc float4 구조체 생성
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
	status = cudaMallocArray(&cudaColorArray, &desc, 256, 1);
	cudaError(status, "colorTable Malloc Error");

	status = cudaMemcpyToArray(cudaColorArray, 0, 0, colorTable, sizeof(float4) * 256, cudaMemcpyHostToDevice);
	cudaError(status, "colorTable Memcpy Error");

	//colorTable을 CUDA 텍스쳐 메모리로 바인딩
	texColorTable.normalized = true;
	texColorTable.filterMode = cudaFilterModeLinear;
	texColorTable.addressMode[0] = cudaAddressModeClamp;
	status = cudaBindTextureToArray(texColorTable, cudaColorArray, desc);
	cudaError(status, "colorTable Bind Texture Error");
}

#endif // !TRASNFERF_UNCTION_H




