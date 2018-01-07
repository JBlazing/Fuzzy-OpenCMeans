#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace utils {

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



void preProcess(cv::Mat & inputImage , cv::Mat & convertedImage , cv::Size & newSize)
{
    inputImage.convertTo(inputImage , CV_32FC3 , 1./255.);
	cv::Mat resize;
    cv::resize(inputImage , resize , newSize);
	cv::cvtColor(resize , resize , cv::COLOR_BGR2Luv);
    inputImage = resize;


	modifyImage(inputImage , convertedImage);

}

void createOutputImgs(cv::Mat image, std::vector<int> &clusterLabels,  cv::Size size, std::vector<cv::Mat> & outImgs)
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
        int cluster = clusterLabels[i];

        cv::MatIterator_<cv::Vec3f> & it = its[cluster];

        for(int j = 0; j < 3 ; j++)
           (*it)[j] = row[j];

        for(auto & itt : its)
            itt++;
    }


}

void getClusterLabels(cv::Mat & U, std::vector<int> & labels )
{
    for(int i = 0 ; i < U.rows ; i++)
    {
        cv::Mat itmU = U.row(i);
        cv::Point minLoc , maxLoc;

        cv::minMaxLoc(itmU , NULL , NULL , &minLoc , &maxLoc );

        labels.push_back(maxLoc.x);

    }
}

void mergeClusters(cv::Mat &data , cv::Mat &mergedClusters  , std::vector<int> & clusterLabels , int whiteCluster)
{
    mergedClusters = cv::Mat::zeros(0 , data.cols , data.type());
    for(int i = 0 ; i < data.rows ; i++)
    {
        if(clusterLabels[i] != whiteCluster){
            mergedClusters.push_back(data.row(i));
        }
    }
}

}
#endif // UTILS_H
