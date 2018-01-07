#ifndef DISTANCEMATRIX_H
#define DISTANCEMATRIX_H

#include <opencv2/core.hpp>
#include <vector>
#include <thread>
#include <iostream>


struct d_Mat{

	std::vector<std::vector<float>> mat;
public:
	d_Mat(int numItems)
	{
	mat.resize(numItems);
	for(int i = 0 ; i < numItems ; i++)
	{
	  mat[i].resize(i , INFINITY);
	}
	}

	float operator()(int i , int j) {

		return (j < i) ?  mat.at(i).at(j) : mat.at(j).at(i);
	}
	void setElement(int i , int j , float value){
		mat[i][j] = value;
	}
	float getElement(int i , int j ){
		return (j < i) ?  mat.at(i).at(j) : mat.at(j).at(i);
	}

};

struct dWorker{
	int start;
	int end;
	d_Mat & dMat;
	cv::Mat & data;
};

class DistanceMatrix
{
	//cv::Mat dMat;
	d_Mat dMat;
    std::vector< dWorker > workers;
public:
    DistanceMatrix(cv::Mat & , int);
    void computeDistanceMatrix();
	d_Mat&  getDistanceMatrix() {return dMat;}
};

#endif // DISTANCEMATRIX_H
