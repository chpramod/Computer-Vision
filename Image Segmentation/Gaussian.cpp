#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>

#define EPSILON 0.1
int HC,HS=20,THRESHOLD=2000,Neighboorhood=800,windowRadius=30;
// #define HS 16  //Spatial bandwidth
// #define HC 32   //Color bandwidth
#define LIMIT 20  
#define ITR 1
// #define S 1  
float m=0.01;    //tradeoff
// #define COLORTHRESHOLD 3500
// #define SPATIALTHRESHOLD 10000
// #define Neighboorhood 500
using namespace cv;
using namespace std;
Mat image;

struct colorSpace{
    float l,u,v,row,col;
};
colorSpace mode[1100][700];
int flag[1100][700]={{0}};
stack<colorSpace> myStack;
// vector< vector<colorSpace> > groups(1000000);
// kernel function
float colorDistance(colorSpace x,colorSpace y){

    return (x.l-y.l)*(x.l-y.l)+(x.u-y.u)*(x.u-y.u)+(x.v-y.v)*(x.v-y.v);
}
float spatialDistance(colorSpace x,colorSpace y){
    return (x.row-y.row)*(x.row-y.row)+(x.col-y.col)*(x.col-y.col);
}
float totalDistance(colorSpace x,colorSpace y){
    float dc = (x.l-y.l)*(x.l-y.l)+(x.u-y.u)*(x.u-y.u)+(x.v-y.v)*(x.v-y.v);
    float ds = (x.row-y.row)*(x.row-y.row)+(x.col-y.col)*(x.col-y.col);
    return dc+m*m*ds;
}
float kernel(int i,int j,colorSpace x){
    colorSpace temp={image.at<Vec3b>(i,j)[0],image.at<Vec3b>(i,j)[1],image.at<Vec3b>(i,j)[2],i,j}; 
    float cDist = colorDistance(temp,x);
    float pDist = spatialDistance(temp,x);
    return exp((int)(-0.5*cDist/HC/HC - 0.5*pDist/HS/HS));
}

colorSpace findMean(colorSpace x){
    colorSpace numerator={0,0,0,0,0};
    float weight,denominator=0;
    for(int i=max(0,(int)x.row-windowRadius);i<min(image.rows,(int)x.row+windowRadius);i++){
        for(int j=max(0,(int)x.col-windowRadius);j<min(image.cols,(int)x.col+windowRadius);j++){
            colorSpace tempPixel={image.at<Vec3b>(i,j)[0],image.at<Vec3b>(i,j)[1],image.at<Vec3b>(i,j)[2],i,j};
            if(totalDistance(x,tempPixel)<Neighboorhood){
                weight=kernel(i,j,x);
                denominator+=weight;
                numerator.l+=(weight*image.at<Vec3b>(i,j)[0]);
                numerator.u+=(weight*image.at<Vec3b>(i,j)[1]);
                numerator.v+=(weight*image.at<Vec3b>(i,j)[2]);
                numerator.row+=(weight*i);
                numerator.col+=(weight*j);
            }
            //printf("distance in window %f\n",totalDistance(x,tempPixel));
        }
    }
    numerator.l/=denominator;
    numerator.u/=denominator;
    numerator.v/=denominator;
    numerator.row/=denominator;
    numerator.col/=denominator;
    return numerator;
}

colorSpace findMode(int r,int c){
    int f=0;
    float shift=100;
    colorSpace convergencePoint={image.at<Vec3b>(r,c)[0],
            image.at<Vec3b>(r,c)[1],image.at<Vec3b>(r,c)[2],r,c};
    colorSpace newMean=convergencePoint;

    while(f<LIMIT && shift>EPSILON){
        //printf("convergencePoint l:%d u:%d v:%d",convergencePoint.l,convergencePoint.u,convergencePoint.v);
       //printf("newMean l:%d u:%d v:%d row:%d col:%d",(int)newMean.l,(int)newMean.u,(int)newMean.v,(int)newMean.row,(int)newMean.col);
        newMean=findMean(convergencePoint);
        //printf("newMean l:%d u:%d v:%d row:%d col:%d\n",(int)newMean.l,(int)newMean.u,(int)newMean.v,(int)newMean.row,(int)newMean.col);
        float minDistance=10000;
        for(int i=max(0,(int)convergencePoint.row-windowRadius);i<min(image.rows,(int)convergencePoint.row+windowRadius);i++){
            for(int j=max(0,(int)convergencePoint.col-windowRadius);j<min(image.cols,(int)convergencePoint.col+windowRadius);j++){
        // for(int i=max(0,(int)newMean.row-windowRadius);i<min(image.rows,(int)newMean.row+windowRadius);i++){
        //     for(int j=max(0,(int)newMean.col-windowRadius);j<min(image.cols,(int)newMean.col+windowRadius);j++){
                colorSpace current={image.at<Vec3b>(i,j)[0],image.at<Vec3b>(i,j)[1],image.at<Vec3b>(i,j)[2],i,j};
                float diff=totalDistance(current,newMean);
                if(minDistance>diff){
                    minDistance=diff;
                    newMean.row=i;
                    newMean.col=j;
                }
            }
        }

        newMean.l=image.at<Vec3b>(newMean.row,newMean.col)[0];
        newMean.u=image.at<Vec3b>(newMean.row,newMean.col)[1];
        newMean.v=image.at<Vec3b>(newMean.row,newMean.col)[2];
        //printf("modifiedNewMean l:%d u:%d v:%d row:%d col:%d\n",(int)newMean.l,(int)newMean.u,(int)newMean.v,(int)newMean.row,(int)newMean.col);
        
        shift=sqrt(totalDistance(newMean,convergencePoint));
        if (flag[(int)newMean.row][(int)newMean.col]==1){
            return mode[(int)newMean.row][(int)newMean.col];
        }
        myStack.push(newMean);
        //printf("shift:%f\n",shift);
        convergencePoint=newMean;
        f++;
    }
    return convergencePoint;
}

int main(int argc, char** argv){
    for(HC=10;HC<=10;HC+=5){
        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                flag[i][j]=0;
            }
        }
    image = imread(argv[1], IMREAD_COLOR);   // Read the file
    if(! image.data ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
                                        
     // image conversion
    cvtColor(image, image,COLOR_BGR2Luv);
    printf("Choose the parameters:\n");
    printf("Neighboorhood Tradeoff(m) HS HC THRESHOLD\n");
    //scanf("%d %f %d %d %d",&Neighboorhood,&m,&HS,&HC,&THRESHOLD);
    printf("Please wait...\n");
    //for(int itr=0;itr<ITR;itr++){
    vector< vector<colorSpace> > groups(1000);
    int numGroups=1,count;
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            if (flag[i][j]==0){
                mode[i][j]=findMode(i,j);
                //printf("mode i:%d j:%d l:%d u:%d v:%d row:%d col:%d\n",i,j,(int)mode[i][j].l,(int)mode[i][j].u,(int)mode[i][j].v,(int)mode[i][j].row,(int)mode[i][j].col);
                while (!myStack.empty()){
                    flag[(int)myStack.top().row][(int)myStack.top().col]=1;
                    //printf("%d %d %d\n",mode[myStack.top().row][myStack.top().col].l,mode[myStack.top().row][myStack.top().col].u,mode[myStack.top().row][myStack.top().col].v);
                    mode[(int)myStack.top().row][(int)myStack.top().col]=mode[i][j];
                    myStack.pop();
                }
            }

            if (i!=0 || j!=0){
                float minDist=THRESHOLD*2,tempDist;
                for (int k=0;k<numGroups;k++){
                    tempDist=totalDistance(mode[i][j],groups[k][0]);
                    //tempspatialDist=spatialDistance(mode[i][j],groups[k][groups[k].size()-1]);
                    //printf("%f\n",tempspatialDist);
                    if (minDist>tempDist){
                        count=k;
                    }
                    minDist=min(minDist,tempDist);
                }
                //printf("minDist %f THRESHOLD %d tempspatialDist %f SPATIALTHRESHOLD %d\n",minDist,COLORTHRESHOLD, tempspatialDist, SPATIALTHRESHOLD);
                if (minDist<THRESHOLD){ //&& tempspatialDist<THRESHOLD){
                    //printf("in if\n");
                  //   image.at<Vec3b>(i,j)[0]=(int)groups[count][0].l;
                  //   image.at<Vec3b>(i,j)[1]=(int)groups[count][0].u;
                  //   image.at<Vec3b>(i,j)[2]=(int)groups[count][0].v;
                  // //  printf("#########");
                  //   groups[count].push_back(mode[i][j]);
                    // if(minDist>SPATIALTHRESHOLD){
                    //     groups[numGroups].push_back(mode[i][j]);
                    //     numGroups++;
                    // }
                    //printf("%d %d %d\n",i,j,numGroups);
                }
                else{
                    //printf("in else\n");
                    groups[numGroups].push_back(mode[i][j]);
                    image.at<Vec3b>(i,j)[0]=(int)groups[numGroups][0].l;
                    image.at<Vec3b>(i,j)[1]=(int)groups[numGroups][0].u;
                    image.at<Vec3b>(i,j)[2]=(int)groups[numGroups][0].v;
                    numGroups++;
                    printf("%d %d %d\n",i,j,numGroups);
                }
            }
            else{
               // printf("#########");
                groups[0].push_back(mode[0][0]);
               // printf("#########");
                image.at<Vec3b>(0,0)[0]=(int)groups[0][0].l;
                image.at<Vec3b>(0,0)[1]=(int)groups[0][0].u;
                image.at<Vec3b>(0,0)[2]=(int)groups[0][0].v;
            }
          //  printf("#########");
        }
    }
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            if (i!=0 || j!=0){
                float minDist=THRESHOLD*2,tempDist;
                for (int k=0;k<numGroups;k++){
                    tempDist=totalDistance(mode[i][j],groups[k][0]);
                    //tempspatialDist=spatialDistance(mode[i][j],groups[k][groups[k].size()-1]);
                    //printf("%f\n",tempspatialDist);
                    if (minDist>tempDist){
                        count=k;
                    }
                    minDist=min(minDist,tempDist);
                }
                //printf("minDist %f COLORTHRESHOLD %d tempspatialDist %f SPATIALTHRESHOLD %d\n",minDist,COLORTHRESHOLD, tempspatialDist, SPATIALTHRESHOLD);
                //if (minDist<COLORTHRESHOLD+(m*m)*SPATIALTHRESHOLD){ //&& tempspatialDist<SPATIALTHRESHOLD){
                    //printf("in if\n");
                    image.at<Vec3b>(i,j)[0]=(int)groups[count][0].l;
                    image.at<Vec3b>(i,j)[1]=(int)groups[count][0].u;
                    image.at<Vec3b>(i,j)[2]=(int)groups[count][0].v;
                  //  printf("#########");
                    groups[count].push_back(mode[i][j]);
                    // if(minDist>SPATIALTHRESHOLD){
                    //     groups[numGroups].push_back(mode[i][j]);
                    //     numGroups++;
                    // }
                    //printf("%d %d %d\n",i,j,numGroups);
                //}
                // else{
                //     //printf("in else\n");
                //     groups[numGroups].push_back(mode[i][j]);
                //     image.at<Vec3b>(i,j)[0]=(int)groups[numGroups][0].l;
                //     image.at<Vec3b>(i,j)[1]=(int)groups[numGroups][0].u;
                //     image.at<Vec3b>(i,j)[2]=(int)groups[numGroups][0].v;
                //     numGroups++;
                //     printf("%d %d %d\n",i,j,numGroups);
                // }
            }
           
          //  printf("#########");
        }
    }    
    
    //Mat imageFinal;                                                                   //500 0 10 20 10000 3500
    // for(int i=0;i<image.rows;i++){
    //     for(int j=0;j<image.cols;j++){
    //         image.at<Vec3b>(i,j)[0]=(int)mode[i][j].l;
    //         image.at<Vec3b>(i,j)[1]=(int)mode[i][j].u;
    //         image.at<Vec3b>(i,j)[2]=(int)mode[i][j].v;
    //     }
    // }
    // for(int i=0;i<numGroups;i++)
    //     printf("i:%d num:%d\n",i,(int)groups[i].size());
    groups.clear();

 
    cvtColor(image, image,COLOR_Luv2BGR);
    char n[100];
    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    sprintf(n,"%s_FLAG_D_%d_m_%.2f_HS_%d_HC_%d_t_%d_f_%d.jpg",argv[1],Neighboorhood,m,HS,HC,THRESHOLD,numGroups);
     imwrite(n,image);
    //imshow( "Display window", image );                   // Show our image inside it.

 }   // waitKey(0);                                          // Wait for a keystroke in the window
    //return 0;
}