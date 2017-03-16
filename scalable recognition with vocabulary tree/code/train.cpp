#include "load.hpp"

Mat databaseTF; 
float idfWeight[1000000];
float normalise[1000000];

float customDistance(Mat a,Mat b){
	float dist=0;
	for (int i=0;i<128;i++){
		dist = dist + (a.at<float>(0,i)-b.at<float>(0,i))*(a.at<float>(0,i)-b.at<float>(0,i));
	}
	dist = (float)sqrt(dist);
	return dist;
}

void distributeFeatures(){
	int count=0;
	//printf("iID:%d\n",imageIndex);
	databaseTF = Mat::zeros(finalCenters.rows,imageIndex+1,CV_32F);
	for (int i=0;i<imageIndex;i++){
		for (int j=count;j<count+dataSpaceImage[i].numFeatures;j++){
			int cid=0,selNode=-1;
			for (int L=0;L<level;L++){
				float minDist = FLT_MAX;
				selNode = -1;
				for (int t=cid;t<cid+k;t++){
					Mat a,b;
					a = dataSpace.row(j);
					b = finalCenters.row(t);
					if (isValid[t]==1){
						float tempDist = customDistance(a,b);
						// printf("########\n");
						// printf("%f ",tempDist);
						if (minDist > tempDist){
							minDist = tempDist;
							selNode = t;
						}
					}
				}
				if (selNode!=-1){
					databaseTF.at<float>(selNode,i)+=1;
					cid=k*(selNode+1);
				}
				else
					printf("weird: %d ",selNode);
			}
			//printf("Image index %d Descriptor Index %d selected Node %d\n", i,j,selNode);
		}
		count +=dataSpaceImage[i].numFeatures;
		//cout<<"hey";
		normalise[i]=0;
	}
	
	// for (int i=0;i<imageIndex;i++)
	// 	cout<<normalise[i];
	// printf("done");
	float finalAns =0;
	for (int i=0;i<finalCenters.rows;i++){
		//float ans  =0;
		for (int j=0;j<imageIndex;j++){
			if (databaseTF.at<float>(i,j)!=0){
				databaseTF.at<float>(i,imageIndex)+=1;
			}
			//ans += databaseTF.at<float>(i,j);
		}
		// printf("numLabels %f\n",ans);
		idfWeight[i]=log((imageIndex+1)/(databaseTF.at<float>(i,imageIndex)+1));
		//printf("%f ##\n",idfWeight[i]);
		//printf("%d %d\n",databaseTF.rows, databaseTF.cols);
		//return;
		//printf("weight:%f ni:%f\n",idfWeight[i],databaseTF.at<float>(i,imageIndex));
		for (int j=0;j<imageIndex;j++){
			// printf("row %d image %d mi = %f idf = %f\n",i,j,databaseTF.at<float>(i,j),idfWeight[i]);
			databaseTF.at<float>(i,j)*=idfWeight[i];
			// printf("row %d image %d mi = %f idf = %f\n",i,j,databaseTF.at<float>(i,j),idfWeight[i]);
		}

	}
	// printf("numLabels %f\n",ans);
	//cout<<"weights\n";
	
	for (int j=0;j<imageIndex;j++){
		for (int i=0;i<finalCenters.rows;i++){
			normalise[j]+=(databaseTF.at<float>(i,j)*databaseTF.at<float>(i,j));
    	}
	}
	for (int i=0;i<imageIndex;i++){
		normalise[i]=(float)sqrt(normalise[i]);
    }
 //    for (int i=0;i<imageIndex;i++){

	// 	// printf("%s:%f\n",dataSpaceImage[i].iName.c_str(),normalise[i]);
	// }
	// printf("done\n");
	// cout<<normalise;
	printf("trained\n");
	return;
}