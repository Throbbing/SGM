#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include <cuda_runtime.h>
#include<vector_types.h>
#include<device_launch_parameters.h>
#include"FreeImage.h"

#define PenaltySmall	10
#define PenaltyLarge	50
#define PATHTOTAL		8
#define PAD16(x)		((x+15)&(~15))
struct f4
{
	f4() { x = y = z = w = 0.f; }
	f4(float _x, float _y, float _z, float _w) :
		x(_x), y(_y), z(_z), w(_w) {}
	float x, y, z, w;
};

int	clampIndex(int x, int Min, int Max)
{
	if (x < Min) return Min;
	else if (x > Max) return Max;
	else return x;
}


f4* loadImage(const std::string& filename,
	int& width, int& height)
{
	FreeImage_Initialise(true);

	FIBITMAP* bmpConverted = nullptr;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	//获取文件类型
	fif = FreeImage_GetFileType(filename.c_str());
	if (fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(filename.c_str());

	if (fif == FIF_UNKNOWN)
	{
		std::cout << "纹理格式未知，无法加载" << std::endl;
		return false;
	}
	if (!FreeImage_FIFSupportsReading(fif))
	{
		std::cout << "纹理格式不被支持，无法加载" << std::endl;
		return false;
	}

	//创建纹理 句柄
	FIBITMAP* dib = FreeImage_Load(fif, filename.c_str());
	if (!dib)
	{
		std::cout << "纹理加载失败！" << std::endl;
		return false;
	}

	//在jmxR中，左上角为纹理(0,0)
	//所以需要倒置纹理

	FreeImage_FlipVertical(dib);

	width = FreeImage_GetWidth(dib);
	height = FreeImage_GetHeight(dib);

	//创建纹理数据结构
	auto data = new f4[width*height];

	RGBQUAD rgb;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			FreeImage_GetPixelColor(dib, x, y, &rgb);

			//rgb中，每个分量为BYTE类型(0~255)
			//将其转换为[0,1]的f32型，并保存在Spectrum中
			float r, g, b;
			r = float(rgb.rgbRed);
			g = float(rgb.rgbGreen);
			b = float(rgb.rgbBlue);
			data[y*width + x] = f4(r, g, b, 0.f);
		}
	}

	//释放句柄
	FreeImage_Unload(dib);

	return data;
}
//8
int*	devPathDir;
int*	devWidth;
int*	devHeight;
int*	devDint;
//width*height
float*	devLeftGray;
//width*height
float*  devRightGray;


//width*height*dint
float*	devImageCost;
float*	devImageInterCost;
//width*height*dint*PathTotal
float*	devPathCost;
//width*height*dint
float*  devImageS;
//width*height*PathTotal
float*	devPathMin;
//width*height
int*    devDisparity;

int		hostWidth;
int		hostHeight;
int		hostDint;
int*	hostDisparity;
f4*		hostLeft;
f4*		hostRight;


__device__ int	cudaClampIndex(int x, int Min, int Max)
{
	if (x < Min) return Min;
	else if (x > Max) return Max;
	else return x;
}

__device__ int	cudaMinI(int a, int b)
{
	return a < b ? a : b;
}

__device__ float cudaMinF(float a, float b)
{
	return a < b ? a : b;
}


__device__ int cudaMaxI(int a, int b)
{
	return a > b ? a : b;
}
__device__ float cudaMaxF(float a, float b)
{
	return a > b ? a : b;
}


__device__ float cudaAbs(float a)
{
	return a > 0.f ? a : -a;
}

__device__ float interHalf(int x, int y, float* image, float iq,
	int width,int height,int dint)
{
	int xplusone = cudaMinI(width - 1, x + 1);
	int xsubone = cudaMaxI(0, x - 1);

	int index0 = y*width + x;
	int index1 = y*width + xplusone;
	int index2 = y*width + xsubone;

	float dpq = cudaMinI(cudaAbs(image[index0] - iq), cudaAbs((image[index0] + image[index1]) / 2 - iq));
	dpq = cudaMinI(dpq, cudaAbs((image[index0] + image[index2]) / 2 - iq));

	return dpq;
}

__global__ void calCost(float* imageCost,float* left,float*right,float* imageS,float* pathCost,float* pathMin,
	int* pwidth,int* pheight,int* pdint)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	int width = *pwidth;
	int height = *pheight;
	int dint = *pdint;

	if (x >= width || y >= height ) return;
	


	//pathRowCost 代表该路径上的L(p,d)   width*height**dint*pathtotal
	//pathRowMinCost 代表该路径上的Lmin(p,d)  width*height*pathtotal
	auto pathCostStride = width*height*dint;
	auto pathRowCost = pathCost;
	auto pathMinStride = width*height;
	auto pathRowMinCost = pathMin;
	float minCost = 1 << 21;

	//初始化pathMin
	auto imageIndex = y*width + x;
	for (int p = 0;p < PATHTOTAL;++p)
	{
		pathRowMinCost = pathMin + p*pathMinStride;
		pathRowMinCost[imageIndex] = 1 << 21;
	}

	
	for (int d = 0;d < dint;++d)
	{
		//计算cost
		auto costIndex = y*width*dint + x*dint + d;
		auto cost = 0.f;
		if (d >= x)
		{
			cost = (float)(1 << 20);
		}
		else
		{
			cost = cudaMinF(interHalf(x, y, left, right[imageIndex - d], width, height, dint),
				interHalf(x - d, y, right, left[imageIndex], width, height, dint));
		}
		imageCost[costIndex] = cost;

		minCost = cudaMinF(minCost, cost);

		//初始化imageS
		imageS[costIndex] = 0;


		//初始化pathCost
		for (int p = 0;p < PATHTOTAL;++p)
		{
			
			pathRowCost = pathCost + p*pathCostStride;
			pathRowCost[costIndex] = 1 << 20;

			
		}

		//pathCost边界判断
		if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
		{
			pathRowCost = pathCost + 0 * pathCostStride;
			pathRowCost[costIndex] = cost;

			pathRowCost = pathCost + 1 * pathCostStride;
			pathRowCost[costIndex] = cost;

			pathRowCost = pathCost + 2 * pathCostStride;
			pathRowCost[costIndex] = cost;

			pathRowCost = pathCost + 3 * pathCostStride;
			pathRowCost[costIndex] = cost;

			pathRowCost = pathCost + 4 * pathCostStride;
			pathRowCost[costIndex] = cost;

			pathRowCost = pathCost + 5 * pathCostStride;
			pathRowCost[costIndex] = cost;

			pathRowCost = pathCost + 6 * pathCostStride;
			pathRowCost[costIndex] = cost;

			pathRowCost = pathCost + 7 * pathCostStride;
			pathRowCost[costIndex] = cost;
			

			imageS[costIndex] += cost * 8;
		}

		
	}//end for d

	//pathMin边界判定
	if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
	{
		pathRowMinCost = pathMin + 0 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;

		pathRowMinCost = pathMin + 1 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;

		pathRowMinCost = pathMin + 2 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;

		pathRowMinCost = pathMin + 3 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;

		pathRowMinCost = pathMin + 4 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;

		pathRowMinCost = pathMin + 5 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;

		pathRowMinCost = pathMin + 6 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;

		pathRowMinCost = pathMin + 7 * pathMinStride;
		pathRowMinCost[imageIndex] = minCost;
	}

	

}

__global__ void subPixelInter(float* imageCost, float* imageInterCost,
	int* pwidth, int* pheight, int* pdint)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int d = threadIdx.z + blockIdx.z*blockDim.z;
	

	int width = *pwidth;
	int height = *pheight;
	int dint = *pdint;
	if (x >= width || y >= height || d >= dint) return;

	auto costIndex = y*width*dint + x*dint + d;

	auto total = 0.f;

	int weight = 0;
	for (int oy = -5;oy < 5;++oy)
	{
		int ny = cudaClampIndex(y + oy, 0, height - 1);
		for (int ox = -5;ox < 5;++ox)
		{
			int nx = cudaClampIndex(x + ox, 0, width - 1);

			int index = ny*width*dint + nx*dint + d;

			total += imageCost[index];
			weight++;
		}
	}

	imageInterCost[costIndex] = total / weight;
		
}

/*
__global__ void updatePath(float* imageCost, float* imageS, float* pathCost,float* pathMin,
	float* pwidth, float* pheight, float* pdint,int x,int* path)
{
	int y = threadIdx.x + blockIdx.x*blockDim.x;
	int p = threadIdx.y;
	int width = *pwidth;
	int height = *pheight;
	int dint = *pdint;
	if (x >= width - 1 || y >= height - 1 || x < 1 || y < 1) return;

	auto selectPathCost = pathCost + p*(width*height*dint);
	auto selectPathMin = pathMin+p*(width*height);


	auto xstep = path[p];
	auto ystep = path[p];
	auto adjx = x + xstep;
	auto adjy = y + ystep;
	auto adjMin = selectPathMin[y*width + x];


	for (int d = 0;d < dint;++d)
	{
		auto costIndex = y*width*dint + x*dint + d;
		auto imageIndex = y*width + x;


		auto dplusone = cudaMinI(d + 1, dint - 1);
		auto dminusone = cudaMaxI(d - 1, 0);

		auto min0 = cudaMinF(selectPathCost[adjy*width*dint + adjx*dint + d],
			selectPathCost[adjy*width*dint + adjx*dint + dplusone] + PenaltySmall);
		auto min1 = cudaMinF(min0, selectPathCost[adjy*width*dint + adjx*dint + dminusone] + PenaltySmall);
		auto min2 = cudaMinF(min1, adjMin + PenaltyLarge);

		auto cost = imageCost[costIndex] + min2 - adjMin;

		selectPathCost[costIndex] = cost;
		if (cost < selectPathMin[imageIndex])
		{
			selectPathMin[imageIndex] = cost;
		}

		imageS[costIndex] += cost;
	}
}


__global__ void updatePath2(float* imageCost, float* imageS, float* pathCost, float* pathMin,
	float* pwidth, float* pheight, float* pdint, int y, int pi, int* path)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int p = threadIdx.y ;
	int width = *pwidth;
	int height = *pheight;
	int dint = *pdint;
	if (x >= width - 1 || y >= height - 1 || x < 1 || y < 1) return;

	auto selectPathCost = pathCost + p*(width*height*dint);
	auto selectPathMin = pathMin + p*(width*height);


	auto xstep = path[p];
	auto ystep = path[p];
	auto adjx = x + xstep;
	auto adjy = y + ystep;
	auto adjMin = selectPathMin[y*width + x];


	for (int d = 0;d < dint;++d)
	{
		auto costIndex = y*width*dint + x*dint + d;
		auto imageIndex = y*width + x;


		auto dplusone = cudaMinI(d + 1, dint - 1);
		auto dminusone = cudaMaxI(d - 1, 0);

		auto min0 = cudaMinF(selectPathCost[adjy*width*dint + adjx*dint + d],
			selectPathCost[adjy*width*dint + adjx*dint + dplusone] + PenaltySmall);
		auto min1 = cudaMinF(min0, selectPathCost[adjy*width*dint + adjx*dint + dminusone] + PenaltySmall);
		auto min2 = cudaMinF(min1, adjMin + PenaltyLarge);

		auto cost = imageCost[costIndex] + min2 - adjMin;

		selectPathCost[costIndex] = cost;
		if (cost < selectPathMin[imageIndex])
		{
			selectPathMin[imageIndex] = cost;
		}

		imageS[costIndex] += cost;
	}
}
*/

__global__ void forwardPass(float* imageCost, float* imageS, float* pathCost, float* pathMin,
	int* pwidth, int* pheight, int* pdint, int x, int* path)
{
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int width = *pwidth;
	int height = *pheight;
	int dint = *pdint;
	if (y == 0 || y == height - 1 || x >= width || y >= height) return;

	auto imageIndex = y*width + x;
	for (int d = 0;d < dint;++d)
	{
		auto costIndex = y*width*dint + x*dint + d;
		//pathRowCost 代表该路径上的L(p,d)   width*height**dint*pathtotal
		//pathRowMinCost 代表该路径上的Lmin(p,d)  width*height*pathtotal
		auto pathCostStride = width*height*dint;
		auto pathRowCost = pathCost;
		auto pathMinStride = width*height;
		auto pathRowMinCost = pathMin;

		//foward阶段 ，前3次路径
		for (int p = 0;p < 3;++p)
		{
			pathRowCost = pathCost + p*pathCostStride;
			pathRowMinCost = pathMin + p*pathMinStride;


			//计算该路径上的相邻点
			auto xstep = path[p * 2];
			auto ystep = path[p * 2 + 1];
			auto adjx = x + xstep;
			auto adjy = y + ystep;
			auto adjMin = pathRowMinCost[adjy*width + adjx];

			auto dplusone = cudaMinI(d + 1, dint - 1);
			auto dminusone = cudaMaxI(d - 1, 0);

			auto min0 = cudaMinF(pathRowCost[adjy*width*dint + adjx*dint + d],
				pathRowCost[adjy*width*dint + adjx*dint + dplusone] + PenaltySmall);
			auto min1 = cudaMinF(min0, pathRowCost[adjy*width*dint + adjx*dint + dminusone] + PenaltySmall);
			auto min2 = cudaMinF(min1, adjMin + PenaltyLarge);

			auto cost = imageCost[costIndex] + min2 - adjMin;

			pathRowCost[costIndex] = cost;
			if (cost < pathRowMinCost[imageIndex])
			{
				pathRowMinCost[imageIndex] = cost;
			}

			imageS[costIndex] += cost;

		}
	}
}


__global__ void backwardPass(float* imageCost, float* imageS, float* pathCost, float* pathMin,
	int* pwidth, int* pheight, int* pdint, int x, int* path)
{
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int width = *pwidth;
	int height = *pheight;
	int dint = *pdint;
	if (y == 0 || y == height - 1 || x >= width || y >= height) return;

	auto imageIndex = y*width + x;
	for (int d = 0;d < dint;++d)
	{
		auto costIndex = y*width*dint + x*dint + d;

		//pathRowCost 代表该路径上的L(p,d)   width*height**dint*pathtotal
		//pathRowMinCost 代表该路径上的Lmin(p,d)  width*height*pathtotal

		auto pathCostStride = width*height*dint;
		auto pathRowCost = pathCost;
		auto pathMinStride = width*height;
		auto pathRowMinCost = pathMin;

		//backword阶段 ,4~7标号的路径
		for (int p = 4;p < 7;++p)
		{
			pathRowCost = pathCost + p*pathCostStride;
			pathRowMinCost = pathMin + p*pathMinStride;


			//计算该路径上的相邻点
			auto xstep = path[p * 2];
			auto ystep = path[p * 2 + 1];
			auto adjx = x + xstep;
			auto adjy = y + ystep;
			auto adjMin = pathRowMinCost[adjy*width + adjx];

			auto dplusone = cudaMinI(d + 1, dint - 1);
			auto dminusone = cudaMaxI(d - 1, 0);

			auto min0 = cudaMinF(pathRowCost[adjy*width*dint + adjx*dint + d],
				pathRowCost[adjy*width*dint + adjx*dint + dplusone] + PenaltySmall);
			auto min1 = cudaMinF(min0, pathRowCost[adjy*width*dint + adjx*dint + dminusone] + PenaltySmall);
			auto min2 = cudaMinF(min1, adjMin + PenaltyLarge);

			auto cost = imageCost[costIndex] + min2 - adjMin;

			pathRowCost[costIndex] = cost;
			if (cost < pathRowMinCost[imageIndex])
			{
				pathRowMinCost[imageIndex] = cost;
			}

			imageS[costIndex] += cost;

		}
	}
}


__global__ void calDisparity(float* imageS, int* disparity,
	int* pwidth,int *pheight,int* pdint)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;


	int width = *pwidth;
	int height = *pheight;
	int dint = *pdint;

	if (x >= width || y >= height) return;
	auto imageIndex = y*width + x;

	float minValue = float(1 << 23);
	int dindex = -1;
	for (int d = 1;d < dint;++d)
	{
		auto costIndex = y*width*dint + x*dint + d;
		auto s = imageS[costIndex];
		if (s < minValue)
		{
			minValue = s;
			dindex = d;
		}
	}
	disparity[imageIndex] = dindex;
}


void	covertGray(f4* left, f4* right,
	float** leftGray, float** rightGray)
{
	int wh = hostWidth*hostHeight;
	*leftGray = new float[wh];
	*rightGray = new float[wh];
	for (int i = 0;i < wh;++i)
	{
		auto c0 = left[i];
		(*leftGray)[i] = 0.212671f*c0.x + 0.715160f*c0.y + 0.072169f*c0.z;
		auto c1 = right[i];
		(*rightGray)[i] = 0.212671f*c1.x + 0.715160f*c1.y + 0.072169f*c1.z;
	}

}


void		createGPUBuffer()
{
	cudaMalloc(&devPathDir, 8 * sizeof(int) * 2);
	cudaMalloc(&devWidth, sizeof(int));
	cudaMalloc(&devHeight, sizeof(int));
	cudaMalloc(&devDint, sizeof(int));
	cudaMalloc(&devLeftGray, hostWidth*hostHeight * sizeof(float));
	cudaMalloc(&devRightGray, hostWidth*hostHeight * sizeof(float));
	cudaMalloc(&devImageCost, hostWidth*hostHeight*hostDint*sizeof(float));
	cudaMalloc(&devImageInterCost, hostWidth*hostHeight*hostDint * sizeof(float));
	cudaMalloc(&devImageS, hostWidth*hostHeight*hostDint*sizeof(float));
	cudaMalloc(&devPathCost, hostWidth*hostHeight*hostDint*PATHTOTAL*sizeof(float));
	cudaMalloc(&devPathMin, hostWidth*hostHeight*PATHTOTAL*sizeof(float));
	cudaMalloc(&devDisparity, hostWidth*hostHeight*sizeof(int));


	//CPU端 对某些Buffer进行初始化
	std::vector<int> pathDir =
	{ -1,-1,
	-1,0,
	-1,+1,
	0, -1,
	+1,-1,
	+1,0,
	+1,+1,
	0,+1 };
	cudaMemcpy(devPathDir, &pathDir[0], PATHTOTAL * 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devWidth, &hostWidth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devHeight, &hostHeight, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devDint, &hostDint, sizeof(int), cudaMemcpyHostToDevice);

	float* leftGray;
	float* rightGray;
	covertGray(hostLeft, hostRight, &leftGray, &rightGray);
	cudaMemcpy(devLeftGray, leftGray, hostWidth*hostHeight * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devRightGray, rightGray, hostWidth*hostHeight * sizeof(float), cudaMemcpyHostToDevice);

	
	std::vector<float> hostMin(hostWidth*hostHeight, (float)(1 << 21));
	auto stride = hostWidth*hostHeight;
	for (int p = 0;p < PATHTOTAL;++p)
	{
		cudaMemcpy(devPathMin + p*stride, &hostMin[0], stride * sizeof(float), cudaMemcpyHostToDevice);
	}

	delete[] leftGray;
	delete[] rightGray;


	hostDisparity = new int[hostWidth*hostHeight];
	
}

void		freeGPUBuffer()
{
	cudaFree(devPathDir);
	cudaFree(devWidth);
	cudaFree(devHeight);
	cudaFree(devDint);
	cudaFree(devLeftGray);
	cudaFree(devRightGray);
	cudaFree(devImageCost);
	cudaFree(devImageInterCost);
	cudaFree(devImageS);
	cudaFree(devPathCost);
	cudaFree(devPathMin);
	cudaFree(devDisparity);
}






void  hostCalCost()
{
	//计算Cost
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	auto width16 = PAD16(hostWidth);
	auto height16 = PAD16(hostHeight);

	dim3 blockNum(width16 / 16, height16 / 16);
	dim3 threadsPerBlock(16, 16);

	calCost<<<blockNum,threadsPerBlock>>>(devImageCost, devLeftGray, devRightGray, devImageS,
		devPathCost, devPathMin,
		devWidth, devHeight, devDint);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float deltaTime;
	cudaEventElapsedTime(&deltaTime, start, stop);
	std::cout << "Cost计算 耗时: " << deltaTime <<"ms"<<std::endl;

}


void hostCalInterCost()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	auto width16 = PAD16(hostWidth);
	auto height16 = PAD16(hostHeight);
	auto dint16 = PAD16(hostDint);

	dim3 blockNum(width16 / 16, height16 / 16, dint16 / 16);
	dim3 threadsPerBlock(16, 16, 16);

	subPixelInter << <blockNum, threadsPerBlock >> > (devImageCost, devImageInterCost,
		devWidth, devHeight, devDint);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float deltaTime;
	cudaEventElapsedTime(&deltaTime, start, stop);
	std::cout << "InterCost计算 耗时: " << deltaTime << "ms" << std::endl;
}
void hostDP(float* devCost)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	auto height16 = PAD16(hostHeight);
	dim3 blockNum(1, height16 / 16);
	dim3 threadsPerBlock(1, 16);
	
	for (int x = 1;x < hostWidth-1;++x)
	{
		
		forwardPass<<<blockNum,threadsPerBlock>>>(devCost, devImageS, devPathCost, devPathMin,
			devWidth, devHeight, devDint,
			x, devPathDir);

		backwardPass << <blockNum, threadsPerBlock >> > (devCost, devImageS, devPathCost, devPathMin,
			devWidth, devHeight, devDint,
			hostWidth - 1 - x, devPathDir);
	}

	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float deltaTime;
	cudaEventElapsedTime(&deltaTime, start, stop);
	std::cout << "DP计算 耗时: " << deltaTime << "ms" << std::endl;
}

void hostCalDisparity()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	auto width16 = PAD16(hostWidth);
	auto height16 = PAD16(hostHeight);

	dim3 blockNum(width16 / 16, height16 / 16);
	dim3 threadsPerBlock(16, 16);

	calDisparity << < blockNum, threadsPerBlock >> > (devImageS, devDisparity,
		devWidth, devHeight, devDint);


	cudaMemcpy(hostDisparity, devDisparity,
		hostWidth*hostHeight * sizeof(float),
		cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float deltaTime;
	cudaEventElapsedTime(&deltaTime, start, stop);
	std::cout << "时差计算 耗时: " << deltaTime << "ms" << std::endl;
}

void hostOutput()
{
	std::ofstream file;
	file.open("Disparity.ppm", std::ios::out);
	file << "P3" << std::endl;
	file << hostWidth*3 << " " << hostHeight << std::endl;
	file << 255 << std::endl;
	for (int y = 0; y < hostHeight;++y)
	{

		for (int x = 0; x < hostWidth; ++x)
		{
			auto index = y*hostWidth + x;
			auto v = (f4)hostLeft[index];// / (float)hostDint *255.f;
			file << clampIndex((int)v.x, 0, 255) << ' ' <<
				clampIndex((int)v.y, 0, 255) << ' ' <<
				clampIndex((int)v.z, 0, 255) << ' ';

		}

		for (int x = 0; x < hostWidth; ++x)
		{
			auto index = y*hostWidth + x;
			auto v = (f4)hostRight[index];// / (float)hostDint *255.f;
			file << clampIndex((int)v.x, 0, 255) << ' ' <<
				clampIndex((int)v.y, 0, 255) << ' ' <<
				clampIndex((int)v.z, 0, 255) << ' ';

		}

		for (int x = 0; x < hostWidth; ++x)
		{
			auto index = y*hostWidth + x;
			float v = (float)hostDisparity[index];// / (float)hostDint *255.f;
			file << clampIndex((int)v, 0, 255) << ' ' <<
				clampIndex((int)v, 0, 255) << ' ' <<
				clampIndex((int)v, 0, 255) << ' ';

		}


		file << std::endl;

	}
	file.close();
}

int main()
{
	//加载图片
	hostLeft = loadImage("images/left.png", hostWidth, hostHeight);
	hostRight = loadImage("images/right.png", hostWidth, hostHeight);
	hostDint = 100;

//	cudaDeviceProp prop;
//	cudaGetDeviceProperties(&prop, 0);

	///////////////////////////////////////////////
	createGPUBuffer();
	///////////////////////////////////////////////

	hostCalCost();
//	hostCalInterCost();
	hostDP(devImageCost);
	hostCalDisparity();
	hostOutput();
	
	
	////////////////////////////////////////////////
	freeGPUBuffer();
	////////////////////////////////////////////////


	system("pause");
	return 0;
	
}

