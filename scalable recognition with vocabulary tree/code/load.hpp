#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

#define featuresLimit 500
#define img_path "./data"
#define query_path "./data"
#define epsilon 0.00001

struct node{
	int id;
	string iName;
	int numFeatures;
	float score;
	float score1;
	float score2;
};

extern Mat finalCenters,dataSpace;
extern int isValid[100000];
extern node dataSpaceImage[100000];
extern int imageIndex;
extern Mat databaseTF;
extern float idfWeight[1000000];
extern float normalise[1000000];
extern int total,total1,total2,k,level;

void extractFeatures();
void distributeFeatures();
void findSimilarImage();
extern float customDistance(Mat a,Mat b);	