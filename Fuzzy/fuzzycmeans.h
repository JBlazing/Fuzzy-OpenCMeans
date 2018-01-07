#ifndef FUZZYCMEANS_H
#define FUZZYCMEANS_H

#include <opencv2/opencv.hpp>
#include <tuple>

class FuzzyCmeans
{
    cv::Mat U;
    cv::Mat C;

    void calcCluster(cv::Mat & , float d_fuzz);
    float updateFuzz(cv::Mat &, float d_fuzz);
    void initUMatrix(cv::Mat &);
public:
    FuzzyCmeans(int numItems , int dimensions ,int numClusters);
    std::tuple<cv::Mat & , cv::Mat &>
            Cluster(cv::Mat &data , float d_fuzz = 2.f, float ep = .01f );
    cv::Mat& getU(){return U;}
};

#endif // FUZZYCMEANS_H
