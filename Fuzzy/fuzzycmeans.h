#ifndef FUZZYCMEANS_H
#define FUZZYCMEANS_H

#include <opencv2/opencv.hpp>

class FuzzyCmeans
{
    cv::Mat U;
    cv::Mat C;

    void calcCluster(cv::Mat & , float d_fuzz);
    float updateFuzz(cv::Mat &, float d_fuzz);
    void initUMatrix(cv::Mat &);
public:
    FuzzyCmeans(int numItems , int dimensions ,int numClusters);
    void Cluster(cv::Mat &data , float d_fuzz, float ep);
    cv::Mat& getU(){return U;}
};

#endif // FUZZYCMEANS_H
