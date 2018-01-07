#ifndef SOM_H
#define SOM_H

#include <opencv2/core.hpp>
#include <vector>
#include <utility>
#include <functional>
#include <numeric>
#include <cmath>
#include <random>

class som
{
	cv::Mat Nuerons;
	cv::Size num_Nuerons;
	cv::Mat & Items;
	std::vector<int> next_item;
	int num_epochs;
	float learning_Rate;
	int epochs;
	int neighborDistance;

	float distanceFunction(cv::Point & winner , int cur , float neighborDistance);
	void updateWeights(int & winner , int item);
	public:
	som(cv::Mat & items , const cv::Size & n_Neurons , int  n_Epochs,  int  in_neightbor_distance, float  l_Rate ) :
		Items(items)
	{
		cv::RNG r = cv::theRNG();
		num_Nuerons = n_Neurons;
		Nuerons = cv::Mat::zeros(  n_Neurons.height * n_Neurons.width  , items.cols , CV_32F );
		num_epochs = n_Epochs;
		learning_Rate = l_Rate;
		neighborDistance = in_neightbor_distance;
		next_item.resize(items.rows);

		for(int i  = 0 ; i < Nuerons.cols ; i++){
			double min , max;
			cv::Mat c = items.col(i);
			cv::minMaxLoc(c , &min , &max);
			r.fill(c , cv::RNG::UNIFORM , min , max);
		}
		std::iota(next_item.begin() , next_item.end() , 0);
	}	void cluster();
};

#endif // SOM_H
