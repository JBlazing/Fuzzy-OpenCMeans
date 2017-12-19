#ifndef FUZZYPCA_H
#define FUZZYPCA_H

#include <opencv2/core.hpp>
#include <vector>
#include <tuple>
typedef cv::Mat item;
class FuzzyPCA
{
    std::vector<std::vector<item>> Uq;
    std::vector<std::vector<item>> Xq;
    cv::Mat data_Norm , U_Norm;
public:
    FuzzyPCA(cv::Mat & data , cv::Mat & U);
    void FuzzyCovariance();
};

#endif // FUZZYPCA_H
