#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include "load.hpp"

using namespace cv;
using namespace std;
int k,level;

int main(int argc, char** argv){
	for(k=5;k<=5;k++){
		for(level=4;level<=4;level++){
			extractFeatures();
			distributeFeatures();
			findSimilarImage();
			int f=imageIndex;
			float accuracy=((float)total)/f,accuracy1=((float)total1)/f,accuracy2=((float)total2)/f;
			printf("images:%d Accuracy:%f Accuracy1:%f Accuracy2:%f for k:%d level:%d featuresLimit:%d\n",
				imageIndex,accuracy,accuracy1,accuracy2,k,level,featuresLimit);
			dataSpace.release();
			finalCenters.release();
			databaseTF.release();
		}
	}
	return 0;
}
