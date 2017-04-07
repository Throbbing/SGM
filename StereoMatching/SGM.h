/**
*　　　　　　　　┏┓　　　┏┓+ +
*　　　　　　　┏┛┻━━━┛┻┓ + +
*　　　　　　　┃　　　　　　　┃ 　
*　　　　　　　┃　　　━　　　┃ ++ + + +
*　　　　　　 ████━████ ┃+
*　　　　　　　┃　　　　　　　┃ +
*　　　　　　　┃　　　┻　　　┃
*　　　　　　　┃　　　　　　　┃ + +
*　　　　　　　┗━┓　　　┏━┛
*　　　　　　　　　┃　　　┃　　　　　　　　　　　
*　　　　　　　　　┃　　　┃ + + + +
*　　　　　　　　　┃　　　┃　　　　Code is far away from bug with the animal protecting　　　　　　　
*　　　　　　　　　┃　　　┃ + 　　　　神兽保佑,代码无bug　　
*　　　　　　　　　┃　　　┃
*　　　　　　　　　┃　　　┃　　+　　　　　　　　　
*　　　　　　　　　┃　 　　┗━━━┓ + +
*　　　　　　　　　┃ 　　　　　　　┣┓
*　　　　　　　　　┃ 　　　　　　　┏┛
*　　　　　　　　　┗┓┓┏━┳┓┏┛ + + + +
*　　　　　　　　　　┃┫┫　┃┫┫
*　　　　　　　　　　┗┻┛　
*/
#pragma once

#include<iostream>
#include<memory>
#include<vector>
#include"Timer.h"
//#include"IPC\IPC.h"



struct float4
{
	float4() { x = y = z = w = 0.f; }
	float4(float _x, float _y, float _z, float _w) :
		x(_x), y(_y), z(_z), w(_w) {}
	float x, y, z, w;
};

class SGM
{
public:
	SGM(float4* left, float4* right,
		int width, int height,
		int dmin,int dmax,
		float penaltySmall,float penaltyLarge);
	~SGM();
	
	std::unique_ptr<int >  getDisparity();

private:
	inline int			   calCostIndex(int x, int y, int d)
	{
		return y*mWidth*mDmax + x*mDmax + d;
	}
	inline int			   calWHIndex(int x, int y)
	{
		return y*mWidth + x;
	}
	inline int			   clampIndex(int x, int Min, int Max)
	{
		if (x < Min) return Min;
		else if (x > Max) return Max;
		else return x;
	}
	template<typename T>	T findMin(T* values, int size)
	{
		auto m = std::numeric_limits<T>::max();
		for (int i = 0;i < size;++i)
		{
			if (values[i] < m)
				m = values[i];
		}
		return m;
	}
	inline bool			   borderValid(int index, int bounder)
	{
		if (index < 0 || index >= bounder)
			return false;

		return true;
	}

	void				   initBuffer();
	void				   resetPathMin();
	void				   coverToGray();
	void				   computeLeftCost();
	float				   interHalf(int x, int y, float* image,float iq);
	void				   updatePath(int x, int y,  int pathIndex,  bool init = false);
	void				   borderPass();
	void				   forwardPass();
	void				   backwardPass();
	std::unique_ptr<int >  performSGM();




	// width*height
	float4*					mLeftImage;
	float4*					mRightImage;
	float*					mLeftGrayImage;
	float*					mRightGrayImage;

	//width*height*dint
	float*					mLeftCost;
	float*					mLeftS;
	
	//width*height*dint*total
	float*					mPathCost;

	//width*height
	float*					mPathMinCost;
	
	
	int						mWidth;
	int						mHeight;

	int						mDmax;
	float					mPenaltySmall;
	float					mPenaltyLarge;

	int					    mPathTotal;
	
	
	Timer					mTimer;
	



};


