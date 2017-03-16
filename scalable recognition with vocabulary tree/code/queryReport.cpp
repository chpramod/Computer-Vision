#include "load.hpp"

Mat queryTF;
float normImg;
int total=0,total1=0,total2=0;
float customColDistance(Mat a,Mat b,int imgID){
	float dist=0;
	for (int i=0;i<a.rows;i++){
		if (normImg!=0 && normalise[imgID]!=0){
			dist += ((a.at<float>(i,0)/normImg-b.at<float>(i,0)/normalise[imgID])*
				(a.at<float>(i,0)/normImg-b.at<float>(i,0)/normalise[imgID]));
		}
		else if (normImg == 0 && normalise[imgID]!=0){
			dist += (b.at<float>(i,0)/normalise[imgID])*
				(b.at<float>(i,0)/normalise[imgID]);
		}
		else if (normImg!=0 && normalise[imgID]==0){
			dist += (a.at<float>(i,0)/normImg)*
				(a.at<float>(i,0)/normImg);
		}
	}
	// printf("%f ",dist);
	dist = (float)sqrt(dist);
	return dist;
}

float normL1(Mat a,Mat b,int imgID){
	float dist=0;
	for (int i=0;i<a.rows;i++){
		if (normImg>epsilon && normalise[imgID]>epsilon){
			dist +=  fabs(a.at<float>(i,0)/normImg-b.at<float>(i,0)/normalise[imgID]);
		}
		else if (normImg<epsilon && normalise[imgID]>epsilon){
			dist += b.at<float>(i,0)/normalise[imgID];
		}
		else if (normImg>epsilon && normalise[imgID]<epsilon){
			dist += a.at<float>(i,0)/normImg;
		}
		// if(normImg>epsilon && normalise[imgID]>epsilon)
		// 	dist += fabs(a.at<float>(i,0)/normImg-b.at<float>(i,0)/normalise[imgID]);
	}
	return dist;
}

float normL2(Mat a,Mat b,int imgID){
	float dist=0;
	for (int i=0;i<a.rows;i++){
		if(normImg>epsilon && normalise[imgID]>epsilon 
			&& a.at<float>(i,0)>epsilon && b.at<float>(i,0)>epsilon)
			dist += a.at<float>(i,0)/normImg*b.at<float>(i,0)/normalise[imgID];
	}
	dist = 2-2*dist;
	return dist;
}

bool sortByScore(const node &lhs, const node &rhs) { return lhs.score < rhs.score; }
bool sortByScore1(const node &lhs, const node &rhs) { return lhs.score1 < rhs.score1; }
bool sortByScore2(const node &lhs, const node &rhs) { return lhs.score2 < rhs.score2; }

void findSimilarImage(){	
	total=total1=total2=0;
	for(int id=0;id<imageIndex;id++){
		vector <node> result;
		normImg=normalise[id];
		queryTF=databaseTF.col(id);
		for (int i=0;i<imageIndex;i++){
			dataSpaceImage[i].score = customColDistance(queryTF,databaseTF.col(i),i);
			dataSpaceImage[i].score1 = normL1(queryTF,databaseTF.col(i),i);
	    	dataSpaceImage[i].score2 = normL2(queryTF,databaseTF.col(i),i);
			result.push_back(dataSpaceImage[i]);
			// printf("i %d Image name : %s score %f\n",i,dataSpaceImage[i].iName.c_str(),dataSpaceImage[i].score);
		}
		int neigh[4];
	    if(id%4==0){
	    	neigh[0]=id+1;neigh[1]=id+2;neigh[2]=id+3;neigh[3]=id;
	    }
	    if(id%4==1){
	    	neigh[0]=id+1;neigh[1]=id+2;neigh[2]=id-1;neigh[3]=id;
	    }
	    if(id%4==2){
	    	neigh[0]=id+1;neigh[1]=id-1;neigh[2]=id-2;neigh[3]=id;
	    }
	    if(id%4==3){
	    	neigh[0]=id-1;neigh[1]=id-2;neigh[2]=id-3;neigh[3]=id;
	    }
		float cmpScore;
	    for(int f=0;f<3;f++){
			if(f==0){
				sort(result.begin(), result.end(), sortByScore);
				cmpScore=result[3].score;
			}
			if(f==1){
				sort(result.begin(), result.end(), sortByScore1);
				cmpScore=result[3].score1;
			}
			if(f==2){
				sort(result.begin(), result.end(), sortByScore2);
				cmpScore=result[3].score2;
			}
	    	//printf("%s %f\n",name,normImg);
	    	//printf("reached %d\n",imageIndex);
	    	int matches=0;
		    for(int n=0;n<4;n++){
		    	if(f==0){
		    		if(dataSpaceImage[neigh[n]].score<=cmpScore)
		    			matches++;
		    	} 
		    	if(f==1){
		    		if(dataSpaceImage[neigh[n]].score1<=cmpScore)
		    			matches++;
		    	}
		    	if(f==2){
		    		if(dataSpaceImage[neigh[n]].score2<=cmpScore)
		    			matches++;
		    	}   		
		    }			
			if(f==0) total+=matches;
			if(f==1) total1+=matches;
			if(f==2) total2+=matches;
			// printf("total:%d total1:%d total2:%d for %s f:%d\n"
			// 	,total,total1,total2,dataSpaceImage[id].iName.c_str(),f);
		}
		result.clear();
	}

		//} 	
		//printf("Matches:%d for %s\n",matches,dataSpaceImage[id].iName.c_str());
		
	return;
}
