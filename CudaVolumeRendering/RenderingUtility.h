#ifndef RENDERING_UTILITY_H
#define RENDERING_UTILITY_H

#include "Global.h"

//���� ��ȿ�� ���� ����
__device__ bool isSamplePossible(float x, float y, float z);
/*
����ȭ �Լ�
������ ��ȿ�� VOLUME ���� t1, t2�� ���
*/
__device__ bool RayBoxCross(float& t1, float& t2, float rsX, float rsY, float rsZ);
/*
CUDATexture�� ����ϱ� ������ ���� ���̻� ������ �ʴ� �Լ�
3���� ���� ���� �Լ�
*/
__device__ int sample(float x, float y, float z, unsigned char* cudavol);




__device__ bool isSamplePossible(float x, float y, float z)
{
	if (x < 0.0 || x > 255.0)
		return false;
	if (y < 0.0 || y > 255.0)
		return false;
	if (z < 0.0 || z > 225.0)
		return false;

	return true;
}
__device__ bool RayBoxCross(float& t1, float& t2, float rsX, float rsY, float rsZ)
{
	float Xmin = FLT_MIN;
	float Xmax = FLT_MAX;
	float Xt1 = FLT_MAX;
	float Xt2 = FLT_MIN;


	float Ymin = FLT_MIN;
	float Ymax = FLT_MAX;
	float Yt1 = FLT_MAX;
	float Yt2 = FLT_MIN;


	float Zmin = FLT_MIN;
	float Zmax = FLT_MAX;
	float Zt1 = FLT_MAX;
	float Zt2 = FLT_MIN;


	if (cc_dir[0] != 0)
	{
		Xt1 = (0 - rsX) / cc_dir[0];
		Xt2 = (255 - rsX) / cc_dir[0];
		Xmin = __min(Xt1, Xt2);
		Xmax = __max(Xt1, Xt2);
	}
	else
	{
		if (isSamplePossible(128, rsY, rsZ) == false)
		{
			return false;
		}
	}

	if (cc_dir[1] != 0)
	{
		Yt1 = (0 - rsY) / cc_dir[1];
		Yt2 = (255 - rsY) / cc_dir[1];
		Ymin = __min(Yt1, Yt2);
		Ymax = __max(Yt1, Yt2);
	}
	else
	{

		if (isSamplePossible(rsX, 128, rsZ) == false)
		{
			return false;
		}
	}

	if (cc_dir[2] != 0)
	{
		Zt1 = (0 - rsZ) / cc_dir[2];
		Zt2 = (255 - rsZ) / cc_dir[2];
		Zmin = __min(Zt1, Zt2);
		Zmax = __max(Zt1, Zt2);
	}
	else
	{
		if (isSamplePossible(rsX, rsY, 112) == false)
		{
			return false;
		}
	}


	t1 = __max(__max(Xmin, Ymin), Zmin);
	t2 = __min(__min(Xmax, Ymax), Zmax);

	//printf("%f %f \n", t1, t2);

	// �� �ڿ� �ִ� ���� �׸��� �ʴ´�.
	// t1�� ������� ���� �� �ڿ� �׸��� �ִٴ°�
	if (t1 < 0)
	{
		t1 = 0;
	}

	return true;
}
__device__ int sample(float x, float y, float z, unsigned char* cudavol)
{
	int ix = x;
	int iy = y;
	int iz = z;


	float deltaX = x - ix;
	float deltaY = y - iy;
	float deltaZ = z - iz;


	float point1 = cudavol[iz * 256 * 256 + iy * 256 + ix];
	float point2 = cudavol[iz * 256 * 256 + iy * 256 + ix + 1];
	float point3 = cudavol[iz * 256 * 256 + (iy + 1) * 256 + ix];
	float point4 = cudavol[iz * 256 * 256 + (iy + 1) * 256 + ix + 1];


	float point5 = cudavol[(iz + 1) * 256 * 256 + iy * 256 + ix];
	float point6 = cudavol[(iz + 1) * 256 * 256 + iy * 256 + ix + 1];
	float point7 = cudavol[(iz + 1) * 256 * 256 + (iy + 1) * 256 + ix];
	float point8 = cudavol[(iz + 1) * 256 * 256 + (iy + 1) * 256 + ix + 1];


	float temp1 = point1 * (1 - deltaX) + point2 * deltaX;
	float temp2 = point3 * (1 - deltaX) + point4 * deltaX;
	float temp3 = temp1 * (1 - deltaY) + temp2 * deltaY;


	float temp4 = point5 * (1 - deltaX) + point6 * deltaX;
	float temp5 = point7 * (1 - deltaX) + point8 * deltaX;
	float temp6 = temp4 * (1 - deltaY) + temp5 * deltaY;


	float result = temp3 * (1 - deltaZ) + temp6 * deltaZ;


	return static_cast<int>(result);
};


#endif // !RENDERING_UTILITY_H
