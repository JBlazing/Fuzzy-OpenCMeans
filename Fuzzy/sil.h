#ifndef SIL_H
#define SIL_H

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
       silCoe = 0.f;
       num = n;
    }
    void computeCoefficient(){
        this->silCoe = (interCluster - intraCluster) / std::max(intraCluster , interCluster);
    }

};

inline bool operator == (silInfo & a , silInfo & b)
{
    return a.num == b.num;
}


class Sil
{
    std::vector< std::list<silInfo> > clusters;

    double computeInterDistance(silInfo & data , std::list<silInfo> & cluster);
    void computeIntraDistance(silInfo & data , std::list<silInfo> & cluster);
public:
    Sil(item & data , cv::Mat & clusterAssignment);
    void computeSil();
    cv::Mat getClusterAverages();
};

#endif // SIL_H
