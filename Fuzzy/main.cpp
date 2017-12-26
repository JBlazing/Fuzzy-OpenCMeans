#include <iostream>
#include <fstream>
#include <string>
#include <regex>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "fuzzycmeans.h"
#include "fuzzypca.h"
//#include "sil.h"
#include "silt.h"
#include <thread>
void parseFile(char * fileName ,cv::Mat & data)
{
    std::ifstream file("iris.data");
    std::string line;
    std::regex re("(\"([^\"]*)\"|([^,]*))(,|$)");
    data = cv::Mat::zeros(0 , 4 , CV_32F);
    cv::Mat temp = cv::Mat::zeros(1 , 4 , CV_32F);
    float * t = temp.ptr<float>();
    if(file.is_open())
    {
        while(getline(file , line))
        {
            std::sregex_iterator next(line.begin() , line.end() , re);
            std::sregex_iterator end;
            int i = 0;
            while(next != end)
            {
                if(i == 4) break;
                std::smatch match = *next;
                //std::cout << match.str() <<'\n';
                std::string m = match.str();
                double tp = std::stof(m , nullptr);
                t[i] = tp;

                //std::cout << tp << ',';
                next++;
                i++;
            }
            data.push_back(temp);
            std::cout << '\n';
        }
    }
    else{

        std::cout << "here" << '\n';
    }

    file.close();

}


void modifyImage(cv::Mat &input , cv::Mat &output)
{


    //calc number of pixels

    int numPixels = input.total();

    output = cv::Mat::zeros(numPixels, input.channels() , CV_32F);


    cv::MatIterator_<cv::Vec3f> it , end;
    it = input.begin<cv::Vec3f>() , end = input.end<cv::Vec3f>();
    for(int i = 0 ; it != end ; i++ , it++)
    {
        float * row = output.ptr<float>(i);
        for(int j = 0 ; j < input.channels() ; j++)
               row[j] = (*it)[j];
    }


}

void createOutputImgs(cv::Mat image, cv::Mat U,  cv::Size size, std::vector<cv::Mat> & outImgs)
{
    std::vector< cv::MatIterator_<cv::Vec3f> > its;
    for(int i = 0 ; i < outImgs.size() ; i++)
    {
        outImgs[i] = cv::Mat::zeros(size  , CV_32FC3);
        its.push_back(outImgs[i].begin<cv::Vec3f>());
    }

    for(int i = 0 ; i < image.rows; i++)
    {
        float *row = image.ptr<float>(i);
        cv::Mat uRow = U.row(i);
        cv::Point cluster;
        cv::minMaxLoc(uRow , NULL , NULL , NULL , &cluster );

        cv::MatIterator_<cv::Vec3f> & it = its[cluster.x];

        for(int j = 0; j < 3 ; j++)
           (*it)[j] = row[j];

        for(auto & itt : its)
            itt++;
    }


}

int main()
{

    cv::Mat Data = cv::imread("socks.jpg"),
            out, conveted, rsize;
    Data.convertTo(Data,  CV_32FC3 , 1.0/255.0);
    cv::Size s(100,100);

    cv::resize(Data , rsize , s , 0 , 0 , 1 );
    cv::cvtColor(rsize , rsize , CV_BGR2HLS );
    modifyImage(rsize , conveted);
    //parseFile("~/iris.data" , Data);

    FuzzyCmeans f(conveted.rows , conveted.cols , 3);

    auto tuple = f.Cluster(conveted);
    cv::Mat C = std::get<0>(tuple),
            U = std::get<1>(tuple);
    std::cout << U << std::endl;
    std::cout << "seperating Clusters" << std::endl;

    silT a(conveted , U , std::thread::hardware_concurrency() );

    a.calcCoefficent();



    //Sil a(conveted , U);
    std::cout << "Clusters Seperated Computing Sil Scores" << std::endl;
    //a.computeSil();
    std::cout <<"Done Calculating avgs" << std::endl;

    cv::Mat avgs = a.getClusterAverages();

    std::cout << avgs << std::endl;
    std::vector< cv::Mat > outImgs;
    outImgs.resize(U.cols);

    createOutputImgs(conveted  , U ,  rsize.size() , outImgs);
    for(int i = 0 ; i < outImgs.size() ; i++){
        cv::cvtColor(outImgs[i] , outImgs[i] , CV_HLS2BGR);
        cv::imshow("Cluster " + i , outImgs[i]);
     }
    cv::waitKey();

}



