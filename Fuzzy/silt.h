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

    std::vector< std::list<silInfo> > *clusters;
    //std::list<silInfo> * curCluster;

    //std::list<silInfo>::iterator curClusterStart , curClusterEnd;
    std::vector < std::list<silInfo>::iterator > clustersStart , clustersEnd;

};

class silT
{

    std::vector< std::list<silInfo> > clusters;


    //std::vector< std::vector<  silWorker >> workers;
    std::vector< silWorker > workers;
public:
    silT(item & data , cv::Mat & clusterAssignment , int numThreads);
    void calcCoefficent();
    cv::Mat getClusterAverages();


};

#endif // SILT_H
