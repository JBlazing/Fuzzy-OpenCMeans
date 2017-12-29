#ifndef SILT_H
#define SILT_H

#include <opencv2/core.hpp>
#include <list>
#include <vector>
#include <cmath>
#include <iostream>


typedef cv::Mat item;


struct silInfo{

    int num;
    item data;
    double intraCluster;
    double interCluster;
    double silCoe;

    silInfo(item &d , int n){
       data = d;
       intraCluster = 0.f;
       interCluster = 0.f;
       silCoe = INFINITY;
       num = n;
    }
    void computeCoefficient(){
        this->silCoe = (interCluster - intraCluster) / std::max(intraCluster , interCluster);
    }

};



struct silWorker{

    int start,
        end,
        numClusters;
    cv::Mat & dHat;
    std::vector<float> &silCoe;
    std::vector<int> &clusterLabels;
};

class silT
{


    std::vector<float> silCoe;
    int nClusters;

    //std::vector< std::vector<  silWorker >> workers;
    std::vector< silWorker > workers;
public:
    silT(cv::Mat & dMatrix, std::vector<int> & clusterLables, int numClusters  ,int numThreads);
    void calcCoefficent();
    cv::Mat getClusterAverages(std::vector<int> & clusterLabels);
    std::vector<float>& getSilCoe(){return silCoe;}


};

#endif // SILT_H
