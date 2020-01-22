
#include "Global.h"
#include "TransferFunction.h"
#include "MIP.h"
#include "CameraMIP.h"
#include "VolumeRendering.h"

#include <chrono>
#include <gl/freeglut.h>


void MyInit();
void FileRead();
void MyDisplay();
void FillMyTexture();



void MyInit()
{
	//gl ���� �ʱ�ȭ
	glClearColor(0.0, 0.0, 0.0, 0.0);

	//���� �б�
	FileRead();

	//TransferFunction ����
	SetTransferFunction();

	//GL �ؽ��ĸ� �ϼ��ϴ� �Լ�
	FillMyTexture();


	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, &MyTexture[0]);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);
}



void MyDisplay()
{


	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_QUADS);
	float fSize = 0.8f;

	glTexCoord2f(0.0, 0.0);
	glVertex3f(-fSize, -fSize, 0.0);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(-fSize, fSize, 0.0);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(fSize, fSize, 0.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(fSize, -fSize, 0.0);

	glEnd();
	glutSwapBuffers();


}



void FillMyTexture()
{
	unsigned char* cudaTexture;
	unsigned char* cudaVol;

	cudaError_t status;

	//CUDA texture ���� �Ҵ�
	status = cudaMalloc((void**)&cudaTexture, sizeof(unsigned char) * WIDTH * HEIGHT * 3);
	cudaError(status, "cudaTexture Malloc Error");

	//Volume�� ������ ���� �Ҵ�
	status = cudaMalloc((void**)&cudaVol, sizeof(unsigned char) * 256 * 256 * 225);
	cudaError(status, "cudaVol Malloc Error");

	//host volume�� Device�� ī��
	status = cudaMemcpy(cudaVol, vol, sizeof(unsigned char) * 256 * 256 * 225, cudaMemcpyHostToDevice);
	cudaError(status, "cudaVol memcpy Error");


	//CUDA Texture�� ����� ���� ����ü ����
	cudaExtent dimSizeVol;
	dimSizeVol.depth = VOL_DEPTH;
	dimSizeVol.height = VOL_HEIGHT;
	dimSizeVol.width = VOL_WIDTH;
	cudaChannelFormatDesc descVol = cudaCreateChannelDesc<unsigned char>();


	//3DArray �Ҵ�
	status = cudaMalloc3DArray(&cuda3DVol, &descVol, dimSizeVol);
	cudaError(status, "3DArray Malloc Error");

	//3DArray �Ӽ��� ����
	cudaMemcpy3DParms volParam = { 0 };
	volParam.srcPtr = make_cudaPitchedPtr(vol, VOL_WIDTH * sizeof(unsigned char), VOL_WIDTH, VOL_HEIGHT);
	volParam.dstArray = cuda3DVol;
	volParam.extent = dimSizeVol;
	volParam.kind = cudaMemcpyHostToDevice;

	//3DArray ������ �Ӽ��� �ѱ��
	status = cudaMemcpy3D(&volParam);
	cudaError(status, "3DArray param memcpy Error");


	//textureVolume �Ӽ��� ����
	texVol.normalized = false;
	texVol.filterMode = cudaFilterModeLinear;
	texVol.addressMode[0] = cudaAddressModeClamp;
	texVol.addressMode[1] = cudaAddressModeClamp;
	texVol.addressMode[2] = cudaAddressModeClamp;


	//3DArray�� CUDA Textrue�� ���ε�
	status = cudaBindTextureToArray(texVol, cuda3DVol, descVol);
	cudaError(status, "3DArray Bind Texture Error");

	//������ �ð� ���
	auto start = std::chrono::high_resolution_clock::now();



	
	// MIP <<< 256, 256 >>> (cudaTexture); 
	// CameraMIP <<< 256, 256 >>> (cudaTexture); 
	VolumeRender <<< 256, 256 >>> (cudaTexture);
	


	status = cudaDeviceSynchronize();
	cudaError(status, "Synchronize Error");

	//������ �ð� ���
	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count()) / 1000000.0 << "ms\n";



	//�� ������� cudaTexture�� GL Texture�� ��ȯ
	status = cudaMemcpy(MyTexture, cudaTexture, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
	cudaError(status, "CUDA Texture to GLTexture Memcpy Error");

}


int main()
{
	InitCamera();
	SelectTransferFunction();
	int a = 1;
	glutInit(&a, NULL);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("openGL Sample Program");

	MyInit();

	glutDisplayFunc(MyDisplay);
	glutMainLoop();
}
