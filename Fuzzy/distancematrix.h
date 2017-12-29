#ifndef DISTANCEMATRIX_H
#define DISTANCEMATRIX_H

#include <opencv2/core.hpp>
#include <vector>
#include <thread>

struct dWorker{
    int start;
    int end;
    cv::Mat & dMat;
    cv::Mat & data;
};

class DistanceMatrix
{
    cv::Mat dMat;
    std::vector< dWorker > workers;
public:
    DistanceMatrix(cv::Mat & , int);
    void computeDistanceMatrix();
    cv::Mat&  getDistanceMatrix() {return dMat;}
};

#endif // DISTANCEMATRIX_H
