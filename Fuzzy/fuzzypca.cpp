#include "fuzzypca.h"
#include <iostream>
#include <utility>
FuzzyPCA::FuzzyPCA(cv::Mat &data , cv::Mat & U)
{
    Uq.resize(U.cols);
    Xq.resize(U.cols);
    data_Norm = data.clone();
    U_Norm = U.clone();

    cv::Mat d_mean , u_mean;
    cv::reduce(data , d_mean , 0 , CV_REDUCE_AVG  );
    cv::reduce(U , u_mean , 0 , CV_REDUCE_AVG );

    std::cout << d_mean << std::endl;

    for(int i = 0 ; i < data.rows ; i++)
    {
        item itm = data_Norm.row(i);
        item itmU = U_Norm.row(i);
        itm -= d_mean;
        u_mean -= u_mean;

        cv::Point minLoc , maxLoc;

        cv::minMaxLoc(itmU , NULL , NULL , &minLoc , &maxLoc );

        Uq[maxLoc.x].push_back(itmU);
        Xq[maxLoc.x].push_back(itm);

    }


}
//Come back to to once this is dealt with in papers
void FuzzyPCA::FuzzyCovariance()
{
    std::vector<cv::Mat> dHat , sStar, ssStar;

    dHat.resize(Xq.size());
    sStar.resize(Uq.size());
    ssStar.resize(Uq.size());





    //compute DHat
    for(int q = 0 ; q < dHat.size() ; q++)
    {
        dHat[q] = cv::Mat::zeros(Xq[q].size() , Xq[q].size() , CV_32F);
        sStar[q] = cv::Mat::zeros(Xq[q].size() , Xq[q].size() , CV_32F);
        ssStar[q] = cv::Mat::zeros(Xq[q].size() , Xq[q].size() , CV_32F);
        //compute d*
        for(int i = 0 ; i < Xq[q].size() ; i++ )
        {
            cv::Mat & Xi = Xq[q][i];
            cv::Mat & Ui = Uq[q][i];
            float * Hat = dHat[q].ptr<float>(i);
            float * s = sStar[q].ptr<float>(i);
            float * ss = ssStar[q].ptr<float>(i);

            for(int j = 0 ; j < Xq[q].size() ; j++){
                cv::Mat & Xj = Xq[q][j];
                cv::Mat & Uj = Uq[q][j];

                cv::Mat subX = Xi - Xj,
                        subU = Ui - Uj;
                cv::pow(subX , 2 , subX);
                cv::pow(subU , 2 , subU);
                Hat[j] = cv::sum(subX)[0] * cv::sum(subU)[0];
                s[j] = Ui.dot(Uj) / (float)(Ui.cols - 1);
                ss[j] = Xi.dot(Xj) / (float)(Xi.cols - 1) ;


            }
        }

    }




}
