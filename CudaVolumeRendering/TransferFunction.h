#ifndef TRASNFERF_UNCTION_H
#define TRASNFERF_UNCTION_H

#include "Global.h"

//TransferFunction���� ����
float aColor[3] = { 0.6, 0.6, 0.6 };
float bColor[3] = { 1.0, 1.0, 1.0 };


//color���̺� ����
void makeColorTable(int a, int b, float acol[3], float bcol[3]);
//alpha���̺� ����
void makeAlphaTable(int a, int b);
//TransferFunction ����
void SetTransferFunction();

void makeColorTable(int a, int b, float acol[3], float bcol[3])
{
	for (int i = 0; i < a; ++i)
	{
		colorTable[i].x = (float)((i * acol[0]) / (a));
		colorTable[i].y = (float)((i * acol[1]) / (a));
		colorTable[i].z = (float)((i * acol[2]) / (a));
	}
	// a ���Խ� 0 
	// b-1 ���Խ� (b - 1 - a) / (b-a) = 1
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
	// a ���Խ� 0 
	// b-1 ���Խ� (b - 1 - a) / (b-a) = 1
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



	//100~140������ ���̰� (���� ���̰�)
	makeColorTable(tableRange[0], tableRange[1], aColor, bColor);
	makeAlphaTable(tableRange[0], tableRange[1]);

	cudaError_t status;

	//CUDA Texture�� ����� ���� Desc float4 ����ü ����
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
	status = cudaMallocArray(&cudaColorArray, &desc, 256, 1);
	cudaError(status, "colorTable Malloc Error");

	status = cudaMemcpyToArray(cudaColorArray, 0, 0, colorTable, sizeof(float4) * 256, cudaMemcpyHostToDevice);
	cudaError(status, "colorTable Memcpy Error");

	//colorTable�� CUDA �ؽ��� �޸𸮷� ���ε�
	texColorTable.normalized = true;
	texColorTable.filterMode = cudaFilterModeLinear;
	texColorTable.addressMode[0] = cudaAddressModeClamp;
	status = cudaBindTextureToArray(texColorTable, cudaColorArray, desc);
	cudaError(status, "colorTable Bind Texture Error");
}

#endif // !TRASNFERF_UNCTION_H




