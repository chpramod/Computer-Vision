#include "load.hpp"

Mat img,desc,dataSpace,finalCenters;
Mat currClusterData[10000000],nextClusterData[10000000];
int isValid[100000],imageIndex;
node dataSpaceImage[100000];

bool sortByName(const node &lhs, const node &rhs) { return lhs.iName < rhs.iName; }

void extractFeatures(){
	DIR* imageDirectory = opendir(img_path);
	if(imageDirectory==NULL){
		printf("Unable to open directory %s\n",img_path);
		return;
	}
	struct dirent *pent = NULL;
	imageIndex=0;
	while((pent=readdir(imageDirectory))!=NULL){
		if (strcmp(pent->d_name, ".") == 0 ||
			strcmp(pent->d_name, "..") == 0) {
			continue;
		}		
		//printf("Extracting features of %s\n",pent->d_name);
		char name[20]=img_path;
		strcat(name,"/");
		strcat(name,pent->d_name);
		//img = imread(name,IMREAD_COLOR);
		
    	dataSpaceImage[imageIndex].id=imageIndex;
    	dataSpaceImage[imageIndex].iName=name;
    	imageIndex++;
    }
    //printf("iID:%d\n",imageIndex);
    sort(dataSpaceImage,dataSpaceImage+imageIndex, sortByName);
    for(int zx=0;zx<imageIndex;zx++){
		dataSpaceImage[zx].id=zx;
		//cout << dataSpaceImage[zx].id << " " <<dataSpaceImage[zx].iName<<"\n";
	}
	//cout<<"**"<<imageIndex<<endl;
    Ptr<SiftFeatureDetector> detector=SiftFeatureDetector::create(featuresLimit,3,0.04,10,1.6);

    for(int i=0;i<imageIndex;i++){
    	img = imread(dataSpaceImage[i].iName.c_str(),IMREAD_COLOR);
		if(! img.data){
        	cout <<  "Could not open or find the image" << std::endl ;
        	return;
    	}
    	vector<KeyPoint> keypoints;    	
    	detector->detectAndCompute(img,Mat(),keypoints,desc,false);
    	dataSpaceImage[i].numFeatures=keypoints.size();
    	
    	//cout << dataSpaceImage[imageIndex].id << " " <<dataSpaceImage[imageIndex].iName<<"\n";
    	
    	if(dataSpace.rows==0)
    		desc.copyTo(dataSpace);
    	else
    		vconcat(dataSpace,desc,dataSpace);
    	//printf("%u %u\n",dataSpace.rows,dataSpace.cols);
    	//printf("Found features\n");
    	keypoints.clear();
    	desc.release();
	}
	
	
	dataSpace.copyTo(nextClusterData[0]);
	//printf("%u %u\n",nextClusterData[0].rows,nextClusterData[0].cols);
	//int numb=0;
	for(int L=0;L<level;L++){
		for(int z=0;z<(int)pow(k,L);z++){
			if(!nextClusterData[z].empty()){
				nextClusterData[z].copyTo(currClusterData[z]);
			}
			//printf("%d %u %u\n",z,nextClusterData[z].rows,nextClusterData[z].cols);
		}
		for(int i=0;i<(int)pow(k,L);i++){
			int num[15]={0},shift;
			Mat clusterLabels,centers;
			for(int idash=0;idash<k;idash++)
				nextClusterData[i*k+idash].release();
			if(currClusterData[i].empty() || currClusterData[i].rows<k){
				if (!currClusterData[i].empty()){
					num[0]=currClusterData[i].rows;
					currClusterData[i].copyTo(nextClusterData[i*k]);
				}
				Mat tmp = Mat::zeros(k,128,CV_32F);
				vconcat(finalCenters,tmp,finalCenters);	
			}
			else{
				kmeans(currClusterData[i],k,clusterLabels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0)
					,1,KMEANS_RANDOM_CENTERS,centers);
				if(finalCenters.rows==0)
					centers.copyTo(finalCenters);
				else
					vconcat(finalCenters,centers,finalCenters);
			//printf("%u %u\n",finalCenters.rows,finalCenters.cols);
				for(int j=0;j<clusterLabels.rows;j++){
					int label=clusterLabels.at<int>(j,0);
					//printf("%d",label);
					//Mat rowCopy = currClusterData[i].row(j).clone();	
					if(L!=level-1)
						nextClusterData[i*k+label].push_back(currClusterData[i].row(j));
					num[label]++;
				}
			}
			shift=((int)pow(k,L+1)-k)/(k-1);
			for(int id=0;id<k;id++){
				if(num[id]>0)
					isValid[shift+i*k+id]=1;
				else
					isValid[shift+i*k+id]=-1;
			}
			currClusterData[i].release();
		}
	}
	// printf("total num %d\n",numb);
	printf("made tree iId:%d ",imageIndex);
	closedir(imageDirectory);
	
}
