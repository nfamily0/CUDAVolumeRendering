#ifndef MIP_H
#define MIP_H

#include "Global.h"

__global__ void MIP(unsigned char* cudaTexture)
{
	unsigned char max = 0;
	int offset = threadIdx.x + blockIdx.x * 256;

	for (int i = 0; i < 225; ++i)
	{
		max = __max(max, tex3D(texVol, blockIdx.x, threadIdx.x, i));
	}
	
	printf("%d\n", tex3D(texVol, blockIdx.x, threadIdx.x, 100));

	cudaTexture[offset * 3 + 0] = max;
	cudaTexture[offset * 3 + 1] = max;
	cudaTexture[offset * 3 + 2] = max;

	

}

#endif // !MIPH
