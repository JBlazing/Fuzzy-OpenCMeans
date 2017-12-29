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

/*
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


}

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
*/


void calcCoe(silWorker & w)
{
    int end = w.end;
    std::vector<float> tmps(w.numClusters , 0);

    std::vector<int> &clusterLabels = w.clusterLabels,
                    counts(w.numClusters , 0);


    for(int i = w.start ; i <= end ; i++ ){
        float * dRow = w.dHat.ptr<float>(i);
        for(int j = 0 ; j < clusterLabels.size() ; j++ )
        {
            if(i != j){
                int c = clusterLabels[j];
                tmps[c] += dRow[j];
                counts[c]++;
            }

        }
        float intraCluster = tmps[clusterLabels[i]] / (float)counts[clusterLabels[i]];
        float interCluster = INFINITY;
        for(int j = 0 ; j < tmps.size() ; j++){
            if(j != clusterLabels[i])
            {
                float tmp = tmps[j] / (float) counts[j];
                interCluster = std::min(interCluster , tmp );

            }
        }
        w.silCoe[i] = (interCluster - intraCluster) / std::max(intraCluster , interCluster);
        memset(&tmps[0] , 0 , tmps.size()* sizeof tmps[0]);
        memset(&counts[0] , 0 , counts.size() * sizeof counts[0]);
    }
}

silT::silT(cv::Mat & dMatrix , std::vector<int> & clusterLabels , int numClusters , int numThreads)
{
    silCoe.resize(clusterLabels.size() , INFINITY);
    workers.reserve(numThreads);
    nClusters = numClusters;

    std::cout << dMatrix.rows << std::endl;
    for(int i = 1 ; i <= numThreads ; i++)
    {
        int start = (i - 1) * dMatrix.rows / numThreads,
            end   =  i * dMatrix.rows / numThreads - 1;
        workers.push_back(silWorker{start , end , numClusters , dMatrix , silCoe , clusterLabels  });
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

cv::Mat silT::getClusterAverages(std::vector<int> & clusterLabels){

    cv::Mat avg =   cv::Mat::zeros(1 , nClusters , CV_32F);
    std::vector<int> counts(nClusters , 0 );
    float * avgs = avg.ptr<float>(0);
    for(int i = 0; i < clusterLabels.size() ; i++)
    {
        counts[clusterLabels[i]]++;
        avgs[clusterLabels[i]] += silCoe[i];
    }
    for(int i = 0 ; i < nClusters ; i++)
        avgs[i] /= (float)counts[i];
    return avg;
}

