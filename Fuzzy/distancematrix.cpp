#include "distancematrix.h"
#include <utility>

DistanceMatrix::DistanceMatrix(cv::Mat & items , int numThreads)
{
    //allocate space for martrix
    dMat = cv::Mat::zeros(items.rows , items.rows , CV_32F);
    //Create Threaded workers
    for(int i = 1 ; i <= numThreads ; i++)
    {
        int first = (i - 1) * dMat.rows / numThreads,
            last  = i * dMat.rows / numThreads - 1;
        workers.push_back(dWorker{first , last , dMat , items});
    }
}


void computeMartix(dWorker & worker)
{
    int end = worker.end;
    cv::Mat & data = worker.data;
    for(int i = worker.start ; i <= end ; i++ )
    {
        float * iPtr = worker.dMat.ptr<float>(i);
        cv::Mat iRow = data.row(i);
        for(int j = 0 ; j < data.rows ; j++ )
        {
            if(i != j){
                cv::Mat jRow = data.row(j);
                iPtr[j] = (float)cv::norm(iRow, jRow);
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
