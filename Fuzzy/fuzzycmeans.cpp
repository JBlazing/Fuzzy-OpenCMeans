#include "fuzzycmeans.h"
#include <cmath>
#include <iostream>
void FuzzyCmeans::initUMatrix(cv::Mat & U)
{
    cv::RNG r = cv::theRNG();
    for(int i = 0 ; i < U.rows ; i++)
    {
        cv::Mat t =  U.row(i);
        r.fill(t , cv::RNG::UNIFORM , 0.0, 1.0);
        t /= cv::sum(t)[0];
    }

}


FuzzyCmeans::FuzzyCmeans(int numItems , int dimensions ,int numClusters)
{
    U = cv::Mat::zeros(numItems , numClusters , CV_32F);
    C = cv::Mat::zeros(numClusters , dimensions , CV_32F);
    initUMatrix(U);
}


void FuzzyCmeans::calcCluster(cv::Mat &data, float d_fuzz)
{
    cv::Mat mTemp;
    cv::pow(U , d_fuzz , mTemp );
    for(int j = 0 ; j < C.rows ; j++){

        cv::Mat tmp = cv::Mat::zeros(1 , C.cols  , CV_32F);
        cv::Mat  uCol = mTemp.col(j);

        for(int i = 0 ; i < U.rows ; i++)
        {

            tmp += uCol.row(i) * data.row(i);

        }
        C.row(j) = tmp / cv::sum(uCol)[0];
    }

    return;
}


float FuzzyCmeans::updateFuzz(cv::Mat &data , float d_fuzz)
{
    float p = 2.0 / (d_fuzz - 1);
    float maxi = -1.0f;
    for(int i = 0 ; i < U.rows ; i++ ){
        float * Ur = U.ptr<float>(i);

        for(int j = 0 ; j < U.cols ; j++){
            cv::Mat Xi = data.row(i);
            float numer = cv::norm(Xi - C.row(j));

            float temp = 0.0f;
            for(int k = 0 ; k < C.rows ; k++)
            {
                float denom = cv::norm(Xi - C.row(k));
                temp += std::pow(numer / denom , p );

            }
            float update = 1.0 / temp;
            float dif = std::fabs(update - Ur[j]);
            maxi = std::max(maxi , dif );
            Ur[j] = update;
        }
    }

    return maxi;

}

std::tuple<cv::Mat ,cv::Mat> FuzzyCmeans::Cluster(cv::Mat &data , float d_fuzz , float ep)
{
    float maxDiff;
    int epochs = 0;
    do{

        calcCluster(data , d_fuzz);
        maxDiff = updateFuzz(data , d_fuzz);
        epochs++;
    }while(std::isgreater(maxDiff , ep));
    return std::make_tuple(C , U);
}
