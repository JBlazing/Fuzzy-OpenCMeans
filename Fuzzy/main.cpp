#include <iostream>
#include <fstream>
#include <string>
#include <regex>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "fuzzycmeans.h"
#include "silt.h"
#include "distancematrix.h"
#include <thread>
#include "utils.h"
#include "som.h"
#define NUMCLUSTERS 3


int main()
{

	cv::Mat Data = cv::imread("multi.jpeg"),
            converted, mergedClusters;
    std::vector< cv::Mat > outImgs;
    std::vector<int> clusterLabels;
	cv::Size newSize(100,100);

    utils::preProcess(Data , converted , newSize);

    FuzzyCmeans fuzzCMeans(converted.rows , converted.cols , NUMCLUSTERS);

    auto tuple = fuzzCMeans.Cluster(converted);
    cv::Mat & C  = std::get<0>(tuple),
            & U = std::get<1>(tuple);


    DistanceMatrix disMat(converted , std::thread::hardware_concurrency());
	disMat.computeDistanceMatrix();
	d_Mat & distanceMatrix = disMat.getDistanceMatrix();


    utils::getClusterLabels(U , clusterLabels );
	silT a(distanceMatrix , clusterLabels, NUMCLUSTERS, std::thread::hardware_concurrency() );

    a.calcCoefficent();

    cv::Mat avgs = a.getClusterAverages(clusterLabels);
    std::cout << avgs << std::endl;

    outImgs.resize(U.cols);

    cv::Point whiteCluster;
    cv::minMaxLoc(avgs , NULL,NULL,NULL , &whiteCluster);

    utils::createOutputImgs(converted  , clusterLabels ,  Data.size() , outImgs);
	utils::mergeClusters(converted , mergedClusters, clusterLabels , whiteCluster.x  );


	FuzzyCmeans cmeansMerged(mergedClusters.rows , mergedClusters.cols , 2);
    auto mTuple = cmeansMerged.Cluster(mergedClusters);
    cv::Mat & C2 = std::get<0>(mTuple),
            & U2 = std::get<1>(mTuple);

    std::vector<int> mergedLabels;

    utils::getClusterLabels(U2 , mergedLabels);
    DistanceMatrix cmeansMergedDis(mergedClusters , std::thread::hardware_concurrency() );
    cmeansMergedDis.computeDistanceMatrix();
	d_Mat & distanceMatrix_m = cmeansMergedDis.getDistanceMatrix();

	silT m(distanceMatrix_m , mergedLabels , 2 , std::thread::hardware_concurrency());
    m.calcCoefficent();
    cv::Mat mAvgs = m.getClusterAverages(mergedLabels);
	std::cout << mAvgs << std::endl;

	for(int i = 0 ; i < outImgs.size() ; i++){
		cv::cvtColor(outImgs[i] , outImgs[i] , cv::COLOR_Luv2BGR);
		cv::imshow("Cluster " + std::to_string(i) , outImgs[i]);
   }


    cv::Mat womboCombo = cv::Mat::zeros(outImgs[0].rows, outImgs[0].cols , CV_32FC3);
    for(int i = 0 ; i < outImgs.size() ; i++){
       if(i != whiteCluster.x) womboCombo += outImgs[i];
    }

	cv::imshow("Cluster Combo" , womboCombo );
	cv::waitKey();





}



