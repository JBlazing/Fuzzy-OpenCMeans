#include "som.h"

#include <iostream>

float som::distanceFunction(cv::Point &  winner, int cur , float neighborDistance){
	cv::Point curLocation(cur / num_Nuerons.height , cur % num_Nuerons.width);
	float dis = (float)cv::norm(curLocation-winner);
	return exp((-pow(dis , 2)) / (2* pow(neighborDistance , 2)));
}

void som::updateWeights(int &winner , int item){

	cv::Mat curItem = Items.row(item);
	cv::Point winnerLoc(winner / num_Nuerons.height , winner % num_Nuerons.width);
	float up = exp((float)-epochs / (float)num_epochs);

	float up_lRate = learning_Rate * up ;
	float up_neighbor = neighborDistance * up;

	for(int j = 0 ; j < Nuerons.rows ; j++){
		cv::Mat curNueron = Nuerons.row(j);
		curNueron += up_lRate * distanceFunction(winnerLoc , j , up_neighbor) * (curItem - curNueron);
	}

}

void som::cluster(){


for(epochs = 1 ; epochs <= num_epochs ; epochs++){
	std::shuffle(next_item.begin() , next_item.end(), std::mt19937{std::random_device{}()} );

	for(int i = 0 ; i < next_item.size() ; i++){
		int winner = 0;
		float minDistance = (float)cv::norm(Items.row(next_item[0]) - Nuerons.row(0));
		for(int j = 1 ; j < Nuerons.rows ; j++){
			float tmp = (float)cv::norm(Items.row(next_item[i]) - Nuerons.row(j));
			if(std::isless(tmp , minDistance)){
				minDistance = tmp;
				winner = j;
			}
	  }
		updateWeights(winner, i);
	}

}
	std::cout << Nuerons << std::endl;
}
