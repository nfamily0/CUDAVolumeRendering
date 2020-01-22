#ifndef VOLUME_RENDERING_H
#define VOLUME_RENDERING_H

#include "Global.h"
#include "RenderingUtility.h"


__global__ void VolumeRender(unsigned char* cudaTexture)//, unsigned char* cudaVol)
{
	float rs[3];

	int s = threadIdx.x;
	int t = blockIdx.x;


	rs[0] = cc_eye[0] + cc_up[0] * (t - 128.f) + cc_cross[0] * (s - 128.f);
	rs[1] = cc_eye[1] + cc_up[1] * (t - 128.f) + cc_cross[1] * (s - 128.f);
	rs[2] = cc_eye[2] + cc_up[2] * (t - 128.f) + cc_cross[2] * (s - 128.f);

	float alphaSum = 0;
	float colorSum[3] = { 0,0,0 };

	float t1 = 0;
	float t2 = 0;

	if (RayBoxCross(t1, t2, rs[0], rs[1], rs[2]) == false)
		return;

	for (float i = t1; i < t2; i += 1.f)
	{

		float x = rs[0] + cc_dir[0] * i;
		float y = rs[1] + cc_dir[1] * i;
		float z = rs[2] + cc_dir[2] * i;



		float den = tex3D(texVol, x, y, z);
		float4 rgba = tex1D(texColorTable, den);

		float newColor[3];
		float newAlpha = rgba.w;//cc_color[den][3];
		newAlpha = 1 - pow(1 - newAlpha, 1);

		newColor[0] = rgba.x;//cc_color[den][0];
		newColor[1] = rgba.y;//cc_color[den][1];
		newColor[2] = rgba.z;//cc_color[den][2];


		//알파가 0이면 앞으로 연산할 필요 없음
		//가속화
		if (newAlpha == 0)
			continue;

		float light[3];
		light[0] = (tex3D(texVol, x + 1, y, z) - tex3D(texVol, x - 1, y, z));
		light[1] = (tex3D(texVol, x, y + 1, z) - tex3D(texVol, x, y - 1, z));
		light[2] = (tex3D(texVol, x, y, z + 1) - tex3D(texVol, x, y, z - 1));
		normalized(light);

		float NL = abs(dot(light, cc_dir));
		float NH = abs(dot(light, cc_dir));


		//if 내적이 음수면 절대값 으로 양수로 바꾼다.
		//if 내적이 음숨ㄴ 0으로 바꾼다.

		//            Ia       Ka         Id       Kd              Is    Ks
		newColor[0] = 0.3 * newColor[0] + 0.5 * newColor[0] * NL + 0.2 * 1 * pow(NH, 5);
		newColor[1] = 0.3 * newColor[1] + 0.5 * newColor[1] * NL + 0.2 * 1 * pow(NH, 5);
		newColor[2] = 0.3 * newColor[2] + 0.5 * newColor[2] * NL + 0.2 * 1 * pow(NH, 5);


		//칼러값이 1을 넘지 않게
		newColor[0] = __min(newColor[0], 1.0);
		newColor[1] = __min(newColor[1], 1.0);
		newColor[2] = __min(newColor[2], 1.0);


		colorSum[0] = colorSum[0] + (1 - alphaSum) * (newColor[0] * newAlpha);
		colorSum[1] = colorSum[1] + (1 - alphaSum) * (newColor[1] * newAlpha);
		colorSum[2] = colorSum[2] + (1 - alphaSum) * (newColor[2] * newAlpha);

		alphaSum = alphaSum + (1 - alphaSum) * newAlpha;
		//알파누적값이 1이되면 더이상 그릴 이유가 없음
		//가속화


		if (alphaSum > 0.99f)
			break;


	}

	cudaTexture[(s + (t * 256)) * 3 + 0] = colorSum[0] * 255;
	cudaTexture[(s + (t * 256)) * 3 + 1] = colorSum[1] * 255;
	cudaTexture[(s + (t * 256)) * 3 + 2] = colorSum[2] * 255;
}

#endif // !VOLUME_RENDERING_H
