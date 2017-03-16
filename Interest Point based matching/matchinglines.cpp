#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

int PATCHSIZE=16;
int MAXPATCH=10;
int THRESHOLD1=60000;
int THRESHOLD2=40000;
#define PI 3.1415
float SCREENING=0.8;
using namespace cv;
using namespace std;

struct pd{
    int delx,dely;
    double theta;
    double magnitude;
}gradient[2001][2001];
struct pt{
    int x,y;
};

struct Descriptor{
    pt intPoint;
    float histogram[16][8];
};
float distance(Descriptor a, Descriptor b){
    double dist=0;
    for(int i=0;i<16;i++){
        for(int j=0;j<8;j++){
            dist+=(a.histogram[i][j]-b.histogram[i][j])*(a.histogram[i][j]-b.histogram[i][j]);
        }
    }
    return sqrt(dist);
}

double lambda[2001][2001];
Mat image_rgb,image_gray,image_rgb1,image_gray1;
int main(int argc, char** argv){
   //for (int THRESHOLD1=50000;THRESHOLD1<=100000;THRESHOLD1+=10000){
    //int THRESHOLD2=THRESHOLD1;
    //for (int PATCHSIZE=2;PATCHSIZE<=16;PATCHSIZE*=2){
    //for (float SCREENING=0.65;SCREENING<=0.86;SCREENING+=0.05){
    vector <pt> interestPoints,pointMatch;
    vector< vector<DMatch> > matchings(1000);
    vector <KeyPoint> keyPoints1,keyPoints2;
    vector <pt> finalInterestPoints1, finalInterestPoints2;
    vector <Descriptor> desc,desc1;
    image_rgb = imread(argv[1], IMREAD_COLOR);   // Read the file
    if(! image_rgb.data ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    cvtColor(image_rgb, image_gray, COLOR_RGB2GRAY);
    int width=image_gray.cols, height=image_gray.rows;
    //printf("%d %d",height,width);
    for(int i=0;i<height;i++){
        gradient[i][0].delx=gradient[i][0].dely=gradient[i][0].theta=gradient[i][0].magnitude=0;
        gradient[i][width-1].delx=gradient[i][width-1].dely=gradient[i][width-1].theta=gradient[i][width-1].magnitude=0;
    }
    for(int i=0;i<width;i++){
        gradient[0][i].delx=gradient[0][i].dely=gradient[0][i].theta=gradient[0][i].magnitude=0;
        gradient[height-1][i].delx=gradient[height-1][i].dely=gradient[height-1][i].theta=gradient[height-1][i].magnitude=0;
    }
    for(int i=1;i<height-1;i++){
        for(int j=1;j<width-1;j++){
                gradient[i][j].delx=image_gray.at<uchar>(i,j+1)-image_gray.at<uchar>(i,j-1);
                gradient[i][j].dely=image_gray.at<uchar>(i-1,j)-image_gray.at<uchar>(i+1,j);
                gradient[i][j].magnitude=sqrt(gradient[i][j].delx*gradient[i][j].delx+gradient[i][j].dely*gradient[i][j].dely);
                if (gradient[i][j].delx!=0){
                    gradient[i][j].theta = atan(gradient[i][j].dely/(double)gradient[i][j].delx)*180/PI;
                    if (gradient[i][j].theta > 0){
                        if (gradient[i][j].delx < 0){
                            gradient[i][j].theta = gradient[i][j].theta + 180;
                        }
                    }
                    else{
                        if (gradient[i][j].dely < 0){
                            gradient[i][j].theta = gradient[i][j].theta + 360;   
                        }
                        if (gradient[i][j].delx < 0){
                            gradient[i][j].theta = gradient[i][j].theta + 180;
                        }   
                    }
                }
                else{
                    if (gradient[i][j].dely > 0)    gradient[i][j].theta = 90;
                    else                            gradient[i][j].theta = 270;
                }
            //printf("%lf\n",gradient[i][j].theta);
        }
        //printf("\n");
    }
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            //printf("HI");
            long long x2=0,xy=0,y2=0;
            for(int row=max(0,i-PATCHSIZE);row<=min(height,i+PATCHSIZE);row++){
                for(int col=max(0,j-PATCHSIZE);col<=min(width,j+PATCHSIZE);col++){
                    x2+=(gradient[row][col].delx*gradient[row][col].delx);
                    xy+=(gradient[row][col].delx*gradient[row][col].dely);
                    y2+=(gradient[row][col].dely*gradient[row][col].dely);
                }
            }
            double lambdatemp=(x2*y2-xy*xy)/(double)(x2+y2);
            //printf("%lld %lld %lld %lf",x2,xy,y2,lambda);
            if(lambdatemp>THRESHOLD1){
                lambda[i][j]=lambdatemp;
                pt current={i,j};
                interestPoints.push_back(current);
       //          if (i<=1200 && i >= 800 && j<=1200 && j>=800)   printf("%d %d %lf\n", i, j, lambda);
                // for(int row=max(0,i-PATCHSIZE);row<min(height,i+PATCHSIZE);row++){
                //  for(int col=max(0,j-PATCHSIZE);col<min(width,j+PATCHSIZE);col++){
                //      image_rgb.at<Vec3b>(row,col)[0]=255;
                //      image_rgb.at<Vec3b>(row,col)[1]=0;
                //      image_rgb.at<Vec3b>(row,col)[2]=0;
            }
            else{
                lambda[i][j]=0;
            }
        }
    }
    while (!interestPoints.empty()){
        pt temp = interestPoints.back();
        int i = temp.x;
        int j = temp.y;
        pt maxpt;
        double locmax=lambda[i][j];
        for(int row=max(0,i-MAXPATCH);row<=min(height,i+MAXPATCH);row++){
            for(int col=max(0,j-MAXPATCH);col<=min(width,j+MAXPATCH);col++){
                if (lambda[row][col] > locmax){
                    locmax=lambda[row][col];
                    //maxpt={row,col};
                }
            }
        }
        if(locmax == lambda[i][j]){
            finalInterestPoints1.push_back(temp);
            KeyPoint tempkey = KeyPoint(temp.y, temp.x,0,-1,0,0,-1);
            keyPoints1.push_back(tempkey);
            // i = temp.x; 
            // j = temp.y;
            // for(int row=max(0,i-PATCHSIZE+1);row<=min(height,i+PATCHSIZE-1);row++){
            //     for(int col=max(0,j-PATCHSIZE+1);col<=min(width,j+PATCHSIZE-1);col++){
            //         image_rgb.at<Vec3b>(row,col)[0]=255;
            //         image_rgb.at<Vec3b>(row,col)[1]=0;
            //         image_rgb.at<Vec3b>(row,col)[2]=0;
            //     }
            // }
        }
        interestPoints.pop_back();
    }
    for (int l=0;l<finalInterestPoints1.size();l++){
        pt temp = finalInterestPoints1[l];
        int i = temp.x;
        int j = temp.y;
        Descriptor longList;
        longList.intPoint.x = temp.x;
        longList.intPoint.y = temp.y;
        for(int i=0;i<16;i++){
            for(int j=0;j<8;j++){
                longList.histogram[i][j]=0;
            }
        }

        for(int row=max(0,i-PATCHSIZE);row<min(height,i-PATCHSIZE/2);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[0][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[1][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[2][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[3][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        for(int row=max(0,i-PATCHSIZE/2);row<min(height,i);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[4][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[5][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[6][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[7][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        for(int row=max(0,i);row<min(height,i+PATCHSIZE/2);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[8][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[9][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[10][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[11][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        for(int row=max(0,i+PATCHSIZE/2);row<min(height,i+PATCHSIZE);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[12][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[13][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[14][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[15][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        desc.push_back(longList);
    }
    // for(int i=0;i<16;i++){
    //         for(int j=0;j<8;j++){
    //             printf("%lf ",desc[2].histogram[i][j]);
    //         }
    //         printf("\n");
    //     }
    //printf("%lu\n",desc[2].histogram[10][5]);
    char n[100];
    Mat img_match1;
    drawKeypoints(image_rgb,keyPoints1,img_match1,Scalar(0,0,255));
    //sprintf(n,"set1/int1/T_%d_PS_%d_pts_%lu.png",THRESHOLD1,PATCHSIZE,desc.size());
    imwrite("intPoint1.jpg",img_match1);
 //    imshow( "Display window", image_rgb );                   
    // waitKey(0);                                          
//################################################################################################################
    image_rgb1 = imread(argv[2], IMREAD_COLOR);   // Read the file
    if(! image_rgb1.data ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    cvtColor(image_rgb1, image_gray1, COLOR_RGB2GRAY);
    width=image_gray1.cols; height=image_gray1.rows;
    //printf("%d %d",height,width);
    for(int i=0;i<height;i++){
        gradient[i][0].delx=gradient[i][0].dely=gradient[i][0].theta=gradient[i][0].magnitude=0;
        gradient[i][width-1].delx=gradient[i][width-1].dely=gradient[i][width-1].theta=gradient[i][width-1].magnitude=0;
    }
    for(int i=0;i<width;i++){
        gradient[0][i].delx=gradient[0][i].dely=gradient[0][i].theta=gradient[0][i].magnitude=0;
        gradient[height-1][i].delx=gradient[height-1][i].dely=gradient[height-1][i].theta=gradient[height-1][i].magnitude=0;
    }
    for(int i=1;i<height-1;i++){
        for(int j=1;j<width-1;j++){
                gradient[i][j].delx=image_gray1.at<uchar>(i,j+1)-image_gray1.at<uchar>(i,j-1);
                gradient[i][j].dely=image_gray1.at<uchar>(i-1,j)-image_gray1.at<uchar>(i+1,j);
                gradient[i][j].magnitude=sqrt(gradient[i][j].delx*gradient[i][j].delx+gradient[i][j].dely*gradient[i][j].dely);
                if (gradient[i][j].delx!=0){
                    gradient[i][j].theta = atan(gradient[i][j].dely/(double)gradient[i][j].delx)*180/PI;
                    if (gradient[i][j].theta > 0){
                        if (gradient[i][j].delx < 0){
                            gradient[i][j].theta = gradient[i][j].theta + 180;
                        }
                    }
                    else{
                        if (gradient[i][j].dely < 0){
                            gradient[i][j].theta = gradient[i][j].theta + 360;   
                        }
                        if (gradient[i][j].delx < 0){
                            gradient[i][j].theta = gradient[i][j].theta + 180;
                        }   
                    }
                }
                else{
                    if (gradient[i][j].dely > 0)    gradient[i][j].theta = 90;
                    else                            gradient[i][j].theta = 270;
                }
            //printf("%lf\n",gradient[i][j].theta);
        }
        //printf("\n");
    }
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            //printf("HI");
            long long x2=0,xy=0,y2=0;
            for(int row=max(0,i-PATCHSIZE);row<=min(height,i+PATCHSIZE);row++){
                for(int col=max(0,j-PATCHSIZE);col<=min(width,j+PATCHSIZE);col++){
                    x2+=(gradient[row][col].delx*gradient[row][col].delx);
                    xy+=(gradient[row][col].delx*gradient[row][col].dely);
                    y2+=(gradient[row][col].dely*gradient[row][col].dely);
                }
            }
            double lambdatemp=(x2*y2-xy*xy)/(double)(x2+y2);
            //printf("%lld %lld %lld %lf",x2,xy,y2,lambda);
            if(lambdatemp>THRESHOLD2){
                lambda[i][j]=lambdatemp;
                pt current={i,j};
                interestPoints.push_back(current);
       //          if (i<=1200 && i >= 800 && j<=1200 && j>=800)   printf("%d %d %lf\n", i, j, lambda);
                // for(int row=max(0,i-PATCHSIZE);row<min(height,i+PATCHSIZE);row++){
                //  for(int col=max(0,j-PATCHSIZE);col<min(width,j+PATCHSIZE);col++){
                //      image_rgb.at<Vec3b>(row,col)[0]=255;
                //      image_rgb.at<Vec3b>(row,col)[1]=0;
                //      image_rgb.at<Vec3b>(row,col)[2]=0;
            }
            else{
                lambda[i][j]=0;
            }
        }
    }
    while (!interestPoints.empty()){
        pt temp = interestPoints.back();
        int i = temp.x;
        int j = temp.y;
        pt maxpt;
        double locmax=lambda[i][j];
        for(int row=max(0,i-MAXPATCH);row<=min(height,i+MAXPATCH);row++){
            for(int col=max(0,j-MAXPATCH);col<=min(width,j+MAXPATCH);col++){
                if (lambda[row][col] > locmax){
                    locmax=lambda[row][col];
                    //maxpt={row,col};
                }
            }
        }
        if(locmax == lambda[i][j]){
            finalInterestPoints2.push_back(temp);
            KeyPoint tempkey = KeyPoint(temp.y, temp.x,0,-1,0,0,-1);
            keyPoints2.push_back(tempkey);
            // i = temp.x; 
            // j = temp.y;
            // for(int row=max(0,i-PATCHSIZE+1);row<=min(height,i+PATCHSIZE-1);row++){
            //     for(int col=max(0,j-PATCHSIZE+1);col<=min(width,j+PATCHSIZE-1);col++){
            //         image_rgb1.at<Vec3b>(row,col)[0]=255;
            //         image_rgb1.at<Vec3b>(row,col)[1]=0;
            //         image_rgb1.at<Vec3b>(row,col)[2]=0;
            //     }
            // }
        }
        interestPoints.pop_back();
    }
    for (int l=0;l<finalInterestPoints2.size();l++){
        pt temp = finalInterestPoints2[l];
        int i = temp.x;
        int j = temp.y;
        Descriptor longList;
        longList.intPoint.x = temp.x;
        longList.intPoint.y = temp.y;
        for(int i=0;i<16;i++){
            for(int j=0;j<8;j++){
                longList.histogram[i][j]=0;
            }
        }

        for(int row=max(0,i-PATCHSIZE);row<min(height,i-PATCHSIZE/2);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[0][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[1][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[2][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[3][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        for(int row=max(0,i-PATCHSIZE/2);row<min(height,i);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[4][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[5][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[6][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[7][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        for(int row=max(0,i);row<min(height,i+PATCHSIZE/2);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[8][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[9][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[10][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[11][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        for(int row=max(0,i+PATCHSIZE/2);row<min(height,i+PATCHSIZE);row++){
            for(int col=max(0,j-PATCHSIZE);col<min(width,j-PATCHSIZE/2);col++){
                longList.histogram[12][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j-PATCHSIZE/2);col<min(width,j);col++){
                longList.histogram[13][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j);col<min(width,j+PATCHSIZE/2);col++){
                longList.histogram[14][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
            for(int col=max(0,j+PATCHSIZE/2);col<min(width,j+PATCHSIZE);col++){
                longList.histogram[15][(int)gradient[row][col].theta/45]+=gradient[row][col].magnitude;
            }
        }
        desc1.push_back(longList);
    }
    // for(int i=0;i<16;i++){
    //         for(int j=0;j<8;j++){
    //             printf("%lf ",desc[2].histogram[i][j]);
    //         }
    //         printf("\n");
    //     }
    //printf("%lu\n",desc[2].histogram[10][5]);
    Mat img_match2;
    drawKeypoints(image_rgb1,keyPoints2,img_match2,Scalar(0,0,255));
    //sprintf(n,"set1/int6/T_%d_PS_%d_pts_%lu.png",THRESHOLD2,PATCHSIZE,desc1.size());
     imwrite("intPoint2.jpg",img_match2);

     for(int i=0;i<desc.size();i++){
        double min1,min2,tmpDist;
        min1=min2=10000;
        pt tmp;tmp.x=i;
        for(int j=0;j<desc1.size();j++){
            tmpDist=0;//distance(desc[i],desc1[j]);
            Descriptor a=desc[i];
            Descriptor b=desc1[j];
            for(int i=0;i<16;i++){
                for(int j=0;j<8;j++){
                    tmpDist+=(a.histogram[i][j]-b.histogram[i][j])*(a.histogram[i][j]-b.histogram[i][j]);
                }
            }
            tmpDist=sqrt(tmpDist);
            if(tmpDist<min1){
                min2=min1;
                min1=tmpDist;
                tmp.y=j;
            }
            else{
                if(tmpDist<min2){
                    min2=tmpDist;
                }
            }
        }
        //printf("%lf  %lf\n",min1,min2);
        if(min1/min2<=SCREENING){
            // DMatch match = DMatch(tmp.x, tmp.y, 1, min1);
            // matchings.push_back(match);
            //printf("%f\n",min1/min2);
            pointMatch.push_back(tmp);
        }
    }
    for (int i=0;i<pointMatch.size();i++){
        pt temp = pointMatch[i];
        DMatch match = DMatch(temp.x, temp.y, 1, 0);
        matchings[i].push_back(match);
    }
    Mat img_match;
    drawMatches(image_rgb, keyPoints1, image_rgb1,keyPoints2,matchings, img_match,Scalar(0,255,0),Scalar(255,0,0));
    //sprintf(n,"set1/match/new/T1_%d_T2_%d_PS_%d_SC_%.2f_NO_%lu.png",THRESHOLD1,THRESHOLD2,PATCHSIZE,SCREENING,pointMatch.size());
    imwrite("outputImage.jpg",img_match);
    printf("%d\n",(int)pointMatch.size());
    // int gridx=2, gridy=1;
    // vector<Mat> vec;
    // vec.push_back(image_rgb);
    // vec.push_back(image_rgb1);
    // Mat res = Mat(900,1800,CV_8UC3);
    // tile(vec,res,gridx,gridy);
    // imwrite("collage.jpg",res);
    // desc.clear();
    // desc1.clear();
    // pointMatch.clear();
    // keyPoints1.clear();
    // keyPoints2.clear();
    // matchings.clear();
    // finalInterestPoints2.clear();
    // finalInterestPoints1.clear();
    // interestPoints.clear();
 
 //    imshow( "Display window", image_rgb );                   
    // waitKey(0);                                          

    return 0;
}
