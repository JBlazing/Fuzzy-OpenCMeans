#include "fuzzycmeans.h"
#include <cmath>
#include <iostream>
void initUMatrix(cv::Mat & U)
{
    cv::RNG rng;

    for(int i = 0 ; i < U.rows ; i++)
    {
        float s = 1.0f;
        float * cRow = U.ptr<float>(i);
        for(int j = 1 ; j < U.cols ; j++)
        {
            float tmp;
            do{
                tmp = rng.uniform(0.f , 1.f);

            }while( std::isless(s - tmp , 0) );

            cRow[j] = tmp;
            s -= tmp;

        }
        cRow[0] = s;
    }

}


FuzzyCmeans::FuzzyCmeans(int numItems , int dimensions ,int numClusters)
{
    U = cv::Mat::zeros(numItems , numClusters , CV_32F);
    C = cv::Mat::zeros(numItems , dimensions , CV_32F);
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

void FuzzyCmeans::Cluster(cv::Mat &data , float d_fuzz , float ep)
{
    float maxDiff;
    do{

        calcCluster(data , d_fuzz);
        maxDiff = updateFuzz(data , d_fuzz);

    }while(std::isgreater(maxDiff , ep));
}
