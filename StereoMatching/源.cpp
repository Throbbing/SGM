#include"SGM.h"


extern float4* loadImage(const std::string& filename,
	int& width, int& height);


int main()
{
	float4* imageLeft, *imageRight;
	int width, height;

	imageLeft = loadImage("images/4left.bmp", width, height);
	imageRight = loadImage("images/4right.bmp", width, height);
	SGM sgm(imageLeft, imageRight, width, height,
		0, 100, 10, 80);

	auto dis = sgm.getDisparity();


	system("pause");
	return 0;

}