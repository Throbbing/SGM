#include<cmath>
#include<algorithm>
#include<limits>
#include<string>
#include<fstream>
#include"FreeImage.h"
#include"SGM.h"


#define TOTALPATH	8
std::vector<int>  pathDir = {
	-1,+0,
	-1,-1,
	+0,-1,
	-1,+1,
	+1,+0,
	+1,-1,
	+0,+1,
	+1,+1 };

float4* loadImage(const std::string& filename,
	int& width,int& height)
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
	auto data = new float4[width*height];

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
			data[y*width + x] = float4(r, g, b,0.f);
		}
	}

	//释放句柄
	FreeImage_Unload(dib);

	return data;
}


SGM::SGM(float4* left, float4* right,
	int width, int height,
	int dmin, int dmax,
	float penaltySmall, float penaltyLarge):
	mLeftImage(left),mRightImage(right),
	mWidth(width),mHeight(height),
	mDmax(dmax),
	mPenaltySmall(penaltySmall),mPenaltyLarge(penaltyLarge)
{
	mLeftGrayImage = new float[width*height];
	mRightGrayImage = new float[width*height];

	mLeftCost = new float[width*height*mDmax];
	mLeftS = new float[width*height*mDmax];

	mPathCost = new float[width*height*mDmax];
	mPathMinCost = new float[width*height];

	initBuffer();
}

SGM::~SGM()
{
	delete[] mLeftGrayImage;
	delete[] mRightGrayImage;
	delete[] mLeftCost;
	delete[] mLeftS;
	
	delete[] mPathCost;
	delete[] mPathMinCost;

}

void SGM::initBuffer()
{
	auto whd = mWidth*mHeight*mDmax;
	for (int i = 0;i < whd;++i)
	{
		mLeftCost[i] = 1 << 20;//std::numeric_limits<float>::max();
		mLeftS[i] = 0;
		mPathCost[i] = 1 << 20;//std::numeric_limits<float>::max();
	}
	auto wh = mWidth*mHeight;
	for (int i = 0;i < wh;++i)
	{
		mPathMinCost[i] = 1 << 21;//std::numeric_limits<float>::max();
	}
}

void SGM::resetPathMin()
{
	for (int y = 1;y < mHeight - 1;++y)
	{
		for (int x = 1;x < mWidth - 1;++x)
		{
			auto whIndex = calWHIndex(x, y);
			mPathMinCost[whIndex] = 1 << 21;//std::numeric_limits<float>::max();
			for (int d = 0;d < mDmax;++d)
			{
				auto costIndex = calCostIndex(x, y, d);
				mPathCost[costIndex] = 1 << 20;
			}
		}
	}
}

void SGM::coverToGray()
{
	for (int i = 0;i < mWidth*mHeight;++i)
	{
		auto c0 = mLeftImage[i];
		mLeftGrayImage[i] = 0.212671f*c0.x + 0.715160f*c0.y + 0.072169f*c0.z;
		auto c1 = mRightImage[i];
		mRightGrayImage[i]= 0.212671f*c1.x + 0.715160f*c1.y + 0.072169f*c1.z;
	}
}

float SGM::interHalf(int x, int y, float* image, float iq)
{
	int xplusone = std::min(mWidth-1,x+1);
	int xsubone = std::max(0, x - 1);

	int index0 = y*mWidth + x;
	int index1 = y*mWidth + xplusone;
	int index2 = y*mWidth + xsubone;

	float dpq = std::min(std::fabs(image[index0] - iq), std::fabs((image[index0] + image[index1]) / 2 - iq));
	dpq = std::min(dpq, std::fabs((image[index0] + image[index2]) / 2 - iq));

	return dpq;
}


void SGM::computeLeftCost()
{
	for (int y = 0;y < mHeight;++y)
	{
		for (int x = 0;x < mDmax;++x)
		{
			auto imageIndex = y*mWidth + x;

			for (int d = 0;d < x;++d)
			{
				auto costIndex = calCostIndex(x, y, d);

				mLeftCost[costIndex] =
					std::min(interHalf(x, y, mLeftGrayImage, mRightGrayImage[imageIndex - d]),
						interHalf(x - d, y, mRightGrayImage, mLeftGrayImage[imageIndex]));

//				mLeftCost[costIndex] = std::fabs(mLeftGrayImage[imageIndex] - mRightGrayImage[imageIndex - d]);
			}


			for (int d = x;d < mDmax;++d)
			{
				auto costIndex = calCostIndex(x, y, d);
				mLeftCost[costIndex] = 1<<20;
			}
		}

		for (int x = mDmax;x < mWidth;++x)
		{
			auto imageIndex = y*mWidth + x;

			for (int d = 0;d < mDmax;++d)
			{
				auto costIndex = calCostIndex(x, y, d);

				mLeftCost[costIndex] =
					std::min(interHalf(x, y, mLeftGrayImage, mRightGrayImage[imageIndex - d]),
						interHalf(x - d, y, mRightGrayImage, mLeftGrayImage[imageIndex]));
//				mLeftCost[costIndex] = std::fabs(mLeftGrayImage[imageIndex] - mRightGrayImage[imageIndex - d]);
			}
		}
	}
}


void SGM::updatePath(int x, int y,  int pathIndex, bool init)
{
	

	if (init)
	{
		for (int d = 0;d < mDmax;++d)
		{
			auto costIndex = calCostIndex(x, y, d);
			auto whIndex = calWHIndex(x, y);
			auto cost = mLeftCost[costIndex];
			mPathCost[costIndex] = cost;
			if (cost < mPathMinCost[whIndex])
			{
				mPathMinCost[whIndex] = cost;
			}
			mLeftS[costIndex] += cost*TOTALPATH;
		}
		return;
	}


	int xStep = pathDir[pathIndex * 2];
	int yStep = pathDir[pathIndex * 2 + 1];
	auto adjx = x + xStep;
	auto adjy = y + yStep;

	auto whIndex = calWHIndex(x, y);
	auto adjMin = mPathMinCost[calWHIndex(adjx, adjy)];

	for (int d = 0;d < mDmax;++d)
	{
		auto costIndex = calCostIndex(x, y, d);
		auto dplusone = std::min(d + 1, mDmax - 1);
		auto dminusone = std::max(d - 1, 0);
		

		auto cost = mLeftCost[costIndex] + std::min(
		{ mPathCost[calCostIndex(adjx,adjy,d)],
		mPathCost[calCostIndex(adjx,adjy,dplusone)] + mPenaltySmall,
		mPathCost[calCostIndex(adjx,adjy,dminusone)] + mPenaltySmall,
		adjMin + mPenaltyLarge }) - adjMin;

		mPathCost[costIndex] = cost;
		if (cost < mPathMinCost[whIndex])
		{
			mPathMinCost[whIndex] = cost;
		}

		mLeftS[costIndex] += cost;
	}

}

void	SGM::borderPass()
{
	for (int x = 0;x < mWidth;++x)
	{
		updatePath(x, 0, 0, true);
		updatePath(x, mHeight - 1, 0, true);
	}
	for (int y = 1;y < mHeight - 1;++y)
	{
		updatePath(0, y, 0, true);
	}
}

void	SGM::forwardPass()
{
	for (int ip = 0;ip < TOTALPATH / 2;++ip)
	{
		for (int y = 1;y < mHeight - 1;++y)
		{
			for (int x = 1;x < mWidth - 1;++x)
			{
				updatePath(x, y, ip);
			}

		}
		resetPathMin();
	}
}

void	SGM::backwardPass()
{
	for (int ip = TOTALPATH / 2;ip < TOTALPATH;++ip)
	{
		for (int y = mHeight - 2;y >= 1;--y)
		{
			for (int x = mWidth - 2;x >= 1;--x)
			{
				updatePath(x, y, ip);
			}
		}
		resetPathMin();
	}
}

std::unique_ptr<int> SGM::performSGM()
{
	//转换灰度值
	mTimer.tick();
	coverToGray();
	mTimer.tick();
	std::cout << "灰度值计算  耗时:" << mTimer.deltaTime() << std::endl;

	//计算C(p,d);
	mTimer.tick();
	computeLeftCost();
	mTimer.tick();
	std::cout << "Cpd计算 耗时:" << mTimer.deltaTime() << std::endl;



	//DP
	mTimer.tick();
	borderPass();
	forwardPass();
	backwardPass();
	mTimer.tick();
	std::cout << "DP  耗时:" << mTimer.deltaTime() << std::endl;

	//计算Disparity
	mTimer.tick();
	std::unique_ptr<int> disparity(new int[mWidth*mHeight]);
	for (int y = 0;y < mHeight;++y)
	{
		for (int x = 0;x < mWidth;++x)
		{
			float minCost = std::numeric_limits<float>::max();
			int minD = -1;
			for (int d = 0;d < mDmax;++d)
			{
				auto index = calCostIndex(x, y, d);
				auto cost = mLeftS[index];
				if (cost < minCost)
				{
					minCost = cost;
					minD = d;
				}
			}

			auto whIndex = calWHIndex(x, y);
			disparity.get()[whIndex] = minD;
		}
	}
	mTimer.tick();
	std::cout << "计算视差图 耗时:" << mTimer.deltaTime() << std::endl;

	return disparity;

}

std::unique_ptr<int> SGM::getDisparity()
{
	auto disparity = performSGM();

	/*
	float4* out = new float4[mWidth*mHeight];

	auto wh = mWidth*mHeight;
	for (int i = 0;i < wh;++i)
	{
		float v = (float)disparity.get()[i] /(float)mDmax *255.f;
		out[i] = float4(v, v, v, v);
	}

	auto ipcMgr = jmxRCore::jmxRIPC::getIPCMgr(JmxRGUID);
	if (ipcMgr)
	{
		ipcMgr->pushMsg(EMsg_2D_Vec4 | EMsg_Image, mWidth, mHeight, 0, mWidth*mHeight * sizeof(float4),
			0, 0, out);
	}

	delete[] out;
	*/

	std::ofstream file;
	file.open("Disparity.ppm", std::ios::out);
	file << "P3" << std::endl;
	file << mWidth * 3 << " " << mHeight << std::endl;
	file << 255 << std::endl;
	for (int y = 0; y < mHeight;++y)
	{

		for (int x = 0; x < mWidth; ++x)
		{
			auto index = y*mWidth + x;
			auto v = (float4)mLeftImage[index];// / (float)hostDint *255.f;
			file << clampIndex((int)v.x, 0, 255) << ' ' <<
				clampIndex((int)v.y, 0, 255) << ' ' <<
				clampIndex((int)v.z, 0, 255) << ' ';

		}

		for (int x = 0; x < mWidth; ++x)
		{
			auto index = y*mWidth + x;
			auto v = (float4)mRightImage[index];// / (float)hostDint *255.f;
			file << clampIndex((int)v.x, 0, 255) << ' ' <<
				clampIndex((int)v.y, 0, 255) << ' ' <<
				clampIndex((int)v.z, 0, 255) << ' ';

		}

		for (int x = 0; x < mWidth; ++x)
		{
			auto index = y*mWidth + x;
			float v = (float)disparity.get()[index];// / (float)hostDint *255.f;
			file << clampIndex((int)v, 0, 255) << ' ' <<
				clampIndex((int)v, 0, 255) << ' ' <<
				clampIndex((int)v, 0, 255) << ' ';

		}


		file << std::endl;

	}
	file.close();

	return disparity;
}