#ifndef CAMERA_MIP_H
#define CAMERA_MIP_H

#include "Global.h"
#include "RenderingUtility.h"


__global__ void CameraMIP(unsigned char* cudaTexture)
{
	float rs[3];

	int s = threadIdx.x;
	int t = blockIdx.x;

	unsigned char max = 0;


	rs[0] = cc_eye[0] + cc_up[0] * (t - 128.f) + cc_cross[0] * (s - 128.f);
	rs[1] = cc_eye[1] + cc_up[1] * (t - 128.f) + cc_cross[1] * (s - 128.f);
	rs[2] = cc_eye[2] + cc_up[2] * (t - 128.f) + cc_cross[2] * (s - 128.f);



	float t1 = 0;
	float t2 = 0;

	if (RayBoxCross(t1, t2, rs[0], rs[1], rs[2]) == false)
		return;

	for (float i = t1; i < t2; i += 1.f)
	{

		float x = rs[0] + cc_dir[0] * i;
		float y = rs[1] + cc_dir[1] * i;
		float z = rs[2] + cc_dir[2] * i;


		if ((x > 255 || x < 0) || (y > 255 || y < 0) || (z > 225 || z < 0))
		{
			continue;
		}

		int den = tex3D(texVol, x, y, z);

		max = __max(max, den);


	}

	cudaTexture[(s + (t * 256)) * 3 + 0] = max;
	cudaTexture[(s + (t * 256)) * 3 + 1] = max;
	cudaTexture[(s + (t * 256)) * 3 + 2] = max;
}

#endif // !CAMERA_MIP_H

