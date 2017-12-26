#include "silt.h"
#include <iterator>
#include <thread>
#include <utility>
double computeInterDistance(silInfo & data , std::list<silInfo> & cluster){
    double acum = 0.;
    for(auto & item : cluster)
    {
        acum += cv::norm(data.data , item.data);
    }
    return acum / (double)cluster.size();

}

void computeIntraDistance(silInfo & data , std::list<silInfo> & cluster){
    double acum = 0.;

    for(auto & item : cluster)
    {
        acum += cv::norm(data.data , item.data);
    }
    data.intraCluster = acum / (double)(cluster.size() - 1);
}

silT::silT(item & data , cv::Mat & clusterAssignment, int numThreads){
    int numCluster = clusterAssignment.cols;
    clusters.resize(numCluster);
    workers.reserve(numThreads);
    std::cout << numCluster << std::endl;
    //init clusters
    for(int i = 0 ; i < data.rows ; i++){
        item d = data.row(i);
        cv::Mat itmU = clusterAssignment.row(i);

        cv::Point minLoc , maxLoc;

        cv::minMaxLoc(itmU , NULL , NULL , &minLoc , &maxLoc );

        clusters[maxLoc.x].push_back(silInfo(d , i));
    }


    for(int i = 1 ; i <= numThreads; i++){
        std::vector < std::list<silInfo>::iterator > clustersStart , clustersEnd;

        for(int c = 0 ; c < clusters.size() ; c++){
            int clusterSize = clusters[c].size();
            int start   =   (i-1) * (clusterSize / numThreads) ,
                end     =   i * clusterSize / numThreads - 1;

            auto s  = clusters[c].begin(),
                 e  = clusters[c].begin();

            std::advance(s , start );
            std::advance(e , end);
            clustersStart.push_back(s);
            clustersEnd.push_back(e);
        }
        workers.push_back(silWorker{&clusters , clustersStart , clustersEnd});
        clustersStart.clear();
        clustersEnd.clear();
    }


    /*
    for(int i = 0 ; i < numCluster ; i++){

        workers[i].resize(numWorkersPerCluster);
        int clusterSize = clusters[i].size();

        for(int j = 1 ; j <= numWorkersPerCluster ; j++){

            int start   =   (j-1) * (clusterSize / numWorkersPerCluster) ,
                end     =   j * clusterSize / numWorkersPerCluster - 1;

            auto s  = clusters[i].begin(),
                 e  = clusters[i].begin();

            std::advance(s , start );
            std::advance(e , end);

            workers[i][j-1] = silWorker{&clusters , &(clusters[i]) , s  ,  e};

        }
    }
    */

}

/*
void calcCoe(silWorker & w)
{
    std::list<silInfo>::iterator it,
                                 end = w.curClusterEnd ;
    int numClusters = (*w.clusters).size();
    std::list<silInfo> &cluster = *w.curCluster;
    for(it = w.curClusterStart ; it != end ; it++){
        computeIntraDistance(*it , cluster);
        double mini = INFINITY;
        for(int i = 0 ; i < numClusters ; i++){
            if(w.curCluster != &((*w.clusters)[i])){
                double tmp = computeInterDistance(*it , (*w.clusters)[i]);
                mini = std::min(mini , tmp);
            }

        }
        (*it).interCluster = mini;
        (*it).computeCoefficient();
    }

}
*/

void calcCoe(silWorker & w)
{
    auto clust = w.clusters->begin();
    for(int i = 0 ; i < w.clustersStart.size() ; i++ , clust++)
    {
        std::list<silInfo>::iterator it = w.clustersStart[i],
                                     end = w.clustersEnd[i];

        for( ; it != end ; it++)
        {
            computeIntraDistance(*it , *clust);
            double mini = INFINITY;
            auto otherClust = w.clusters->begin();
            for( ; otherClust != w.clusters->end() ; otherClust++){
                if( clust != otherClust ){
                    double tmp = computeInterDistance(*it , *otherClust);
                    mini = std::min(mini , tmp);
                }
                (*it).interCluster = mini;
                (*it).computeCoefficient();
            }
        }
    }
}

void silT::calcCoefficent()
{
    std::vector<std::thread> threads;

    for(auto & worker : workers){

            threads.push_back(std::thread(calcCoe , std::ref(worker)));

    }


    for(auto & thread : threads){
        thread.join();
    }

}
cv::Mat silT::getClusterAverages(){

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

