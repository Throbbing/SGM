#include"Timer.h"

Timer::Timer()
{
	mIsStop = false;
	mInitTime = 0;
	mStopTime = 0;
	mStartTime = 0;
	mCurrTime = 0;
	mLastTime = 0;
	mPausedTime = 0;
	__int64 frequency;
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
	mSecondsPerCount = 1.0 / (double)frequency;
}

Timer::~Timer()
{
}

void Timer::tick()
{
	if (mIsStop)
	{
		mDeltaTime = 0;
		return;
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&mCurrTime);

	mDeltaTime = (mCurrTime - mLastTime)*mSecondsPerCount;

	mLastTime = mCurrTime;
	if (mDeltaTime < 0)
		mDeltaTime = 0;
}

void Timer::reset()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&mCurrTime);

	mInitTime = mCurrTime;
	mLastTime = mCurrTime;
	mStopTime = 0;
	//mPausedTime = 0;
	mIsStop = false;
	

}

void Timer::stop()
{
	if (!mIsStop)
	{
		mIsStop = true;
		__int64 currTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

		mStopTime = currTime;
	}
}

void Timer::start()
{
	if (mIsStop)
	{
		mIsStop = false;
		QueryPerformanceCounter((LARGE_INTEGER*)&mStartTime);

		mPausedTime += (mStartTime - mStopTime);
		mStopTime = 0;
		mLastTime = mStartTime;


		
	}
}

float Timer::totalTime()
{
	if (mIsStop)
	{
		return (float)((mStopTime - mPausedTime - mInitTime)*mSecondsPerCount);
	}
	else
		return (float)((mCurrTime-mPausedTime-mInitTime)*mSecondsPerCount);


}