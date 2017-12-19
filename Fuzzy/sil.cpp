#include "sil.h"
#include <iostream>
Sil::Sil(item & data , cv::Mat & clusterAssignment)
{
    int numCluster = clusterAssignment.cols;
    clusters.resize(numCluster);

    //init clusters
    for(int i = 0 ; i < data.rows ; i++){
        item d = data.row(i);
        cv::Mat itmU = clusterAssignment.row(i);

        cv::Point minLoc , maxLoc;

        cv::minMaxLoc(itmU , NULL , NULL , &minLoc , &maxLoc );

        clusters[maxLoc.x].push_back(silInfo(d , i));
    }

}



void Sil::computeSil()
{
    std::vector<std::list<silInfo>>::iterator curCluster , otherClusters;

    //for each item compute
    for(curCluster = clusters.begin() ; curCluster != clusters.end() ; curCluster++)
    {
        for(auto & i : *curCluster)
        {
            //compute intra cluster distance for each item in each cluster
            std::list<silInfo> & c = *curCluster;
            computeIntraDistance(i , c);

            double mini = INFINITY;
            for(otherClusters = clusters.begin() ; otherClusters != clusters.end() ; otherClusters++)
            {
                if(curCluster != otherClusters)
                {
                    double dis = computeInterDistance(i , *otherClusters);
                    mini = std::min(mini , dis);
                }
            }
            i.interCluster = mini;
            i.computeCoefficient();

        }
    }

}

double Sil::computeInterDistance(silInfo & data , std::list<silInfo> & cluster){
    double acum = 0.;
    for(auto & item : cluster)
    {
        acum += cv::norm(data.data , item.data);
    }
    return acum / (double)cluster.size();

}

void Sil::computeIntraDistance(silInfo & data , std::list<silInfo> & cluster){
    double acum = 0.;

    for(auto & item : cluster)
    {
        if(!(data == item))
        {
            acum += cv::norm(data.data , item.data);
        }
    }
    data.intraCluster = acum / (double)(cluster.size() - 1);
}


cv::Mat Sil::getClusterAverages(){

    cv::Mat avg =   cv::Mat::zeros(1 , clusters.size() , CV_32F);
    float * avgs = avg.ptr<float>(0);
    for(int i = 0; i < clusters.size() ; i++)
    {
        for(auto & s : clusters[i])
        {
           avgs[i] += s.silCoe;
        }
        avgs[i] /= (float)clusters[i].size();
    }
    return avg;
}
