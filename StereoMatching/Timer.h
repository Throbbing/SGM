#ifndef Timer_H_
#define Timer_H_
#include<iostream>
#include<windows.h>
using namespace std;

class Timer
{
public:
	Timer();
	~Timer();


	void reset();
	void tick();
	void stop();
	float deltaTime()
	{
		return (float)mDeltaTime;
	}
	void start();
	float totalTime();
	

private:
	__int64 mInitTime;//初始时间
	__int64 mStopTime;//暂停时间
	__int64 mStartTime;//重新开始的时间

	__int64 mPausedTime;//暂定过的时间

	double mSecondsPerCount;
	double mDeltaTime;//时间间隔

	__int64 mCurrTime;//
	__int64 mLastTime;

	bool mIsStop;
	

};



#endif // !Timer_H_
