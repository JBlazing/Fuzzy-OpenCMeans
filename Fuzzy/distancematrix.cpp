#include "distancematrix.h"
#include <utility>

DistanceMatrix::DistanceMatrix(cv::Mat & items , int numThreads) :
	dMat(items.rows)
{
    //allocate space for martrix
	//dMat = cv::Mat::zeros(items.rows , items.rows , CV_32F);

	//Create Threaded workers
    for(int i = 1 ; i <= numThreads ; i++)
    {
		int first = (i - 1) * items.rows / numThreads,
			last  = i * items.rows / numThreads - 1;
        workers.push_back(dWorker{first , last , dMat , items});
    }
}


void computeMartix(dWorker & worker)
{
    int end = worker.end;
    cv::Mat & data = worker.data;
	d_Mat & dMat = worker.dMat;
    for(int i = worker.start ; i <= end ; i++ )
    {
        cv::Mat iRow = data.row(i);
		for(int j = 0 ; j < i ; j++ )
        {
            if(i != j){
                cv::Mat jRow = data.row(j);
				float tmp =  (float)cv::norm(iRow, jRow);
				dMat.setElement(i,j,tmp);
            }

        }
    }
}

void DistanceMatrix::computeDistanceMatrix(){

    std::vector<std::thread> threads;
    threads.reserve(workers.size());

    for(auto & worker  : workers){
        threads.push_back(std::thread(computeMartix , std::ref(worker) ));
    }

    for(auto &thread: threads)
    {
        thread.join();

    }


}
