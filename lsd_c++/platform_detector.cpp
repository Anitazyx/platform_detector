#include <stdio.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <cv.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "math.h"

using namespace cv;
using namespace std;


//读取指定文件下的所有图片
vector<Mat> read_images_in_folder(cv::String pattern)
{
 vector<cv::String> fn;
 glob(pattern, fn, false);
 vector<Mat> images;

 size_t count = fn.size(); //number of png files in images folder
 for (size_t i = 0; i < count; i++)
 {
    //  cout << i;
     images.push_back(imread(fn[i])); //直读取图片并返回Mat类型
 }
 return images;
}
 
//图片写入指定文件夹下
void write_images_to_folder(vector<Mat> images_lsd, cv:: String pic_out){
    for(int i = 0; i<images_lsd.size(); i++){
        stringstream wname;
        wname << pic_out << i+1 << ".jpg";
        imwrite(wname.str(), images_lsd[i]);
    }
}


// clipHistPercent 剪枝（剪去总像素的多少百分比）
// histSize 最后将所有的灰度值归到多大的范围
// lowhist 最小的灰度值
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0, int histSize = 255, int lowhist = 0)
{
    
    
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));
    
    float alpha, beta;
    double minGray = 0, maxGray = 0;
    
    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
    
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
    
        cv::Mat hist; //the grayscale histogram
        
        float range[] = {
     0, 256 };
        const float* histRange = {
     range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);
        
        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
    
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }
        
        // locate points that cuts at required value
        float max = accumulator.back();
        
        int clipHistPercent2;
        clipHistPercent2 = clipHistPercent * (max / 100.0); //make percent as absolute
        clipHistPercent2 /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent2)
            minGray++;
        
        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent2))
            maxGray--;
    }
    
    // current range
    float inputRange = maxGray - minGray;
    
    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha + lowhist;             // beta shifts current range so that minGray will go to 0
    
    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);
    
    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
    
        int from_to[] = {
     3, 3};
        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
    }
}



// 纵坐标区域覆盖率
double merge_ratio(double ymin, double ymax, vector<vector<double> >& intervals){
    if (intervals.size() == 0) {
        return 0;
    }
    sort(intervals.begin(), intervals.end());
    vector<vector<int> > merged;
    for (int i = 0; i < intervals.size(); ++i) {
        int L = intervals[i][0], R = intervals[i][1];
        // cout<<"L,R"<<L<<" "<<R<<endl;
        vector<int> a;
        a.push_back(L);
        a.push_back(R);
        if (!merged.size() || merged.back()[1] < L) {
            merged.push_back(a);

        }
        else {
            merged.back()[1] = max(merged.back()[1], R);
        }
    }
   
    int length = 0;
   
    for (int i = 0; i<merged.size(); i++){
        length += merged[i][1]-merged[i][0];
        // cout<<"merge添加了"<<merged[i][1]<<" "<<merged[i][0]<<endl;
    }
    // cout<<"mergesize:"<<merged.size()<<endl;
    // cout<<"length:"<<length<<endl;

    return length/(ymax-ymin);
}


//图像预处理与LSD检测
vector<Mat> lsd_detector_unnormal(vector<Mat> images, int x1, int x2, int y1, int y2, int bias_x){
    int num_total = 0;
    int num_unnormal = 0;
    int num_light = 0;
    int width = x2-x1; 
    int height = y2-y1;
    Rect area(x1, y1, width, height); //需要裁减的矩形区域
    vector<Mat> images_lsd;
      
    double start = double(getTickCount());
    for (int i = 0; i < images.size(); i++){
        num_total ++;

        Mat crop_1 = images[i](area);//裁减
        Mat crop;
        BrightnessAndContrastAuto(crop_1, crop);
        circle(crop, Point(bias_x, 0), 1, Scalar(0, 0, 255), 4, 8);
        circle(crop, Point(x2-x1-bias_x, 0), 1, Scalar(0, 0, 255), 4, 8);
        circle(crop, Point(bias_x, y2-y1), 1, Scalar(0, 0, 255), 4, 8);
        circle(crop, Point(x2-x1-bias_x, y2-y1), 1, Scalar(0, 0, 255), 4, 8);
        imwrite("draw.jpg", crop);
        // num_light += get_lightness(crop);

        // if(get_lightness(crop<10)){
        //     crop = aug(crop);
        // }  

        // if(get_lightness(dst>50)){
        //     crop = contrastEnhance(crop);
        // }
        Mat img;
        cvtColor(crop,img,CV_BGR2GRAY);

    
        Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_NONE); 
        vector<Vec4i> dlines; // 或vector<Vec4f>lines; 
        lsd->detect(img,dlines);
        // Show found lines
        
        // Mat drawnLines(img); //深拷贝，drawnLines和image位于不同内存
        // drawnLines = Scalar(0, 0, 0);
        if (dlines.size()!=0){

            vector<vector<double> > list_keypoint;
            for (int j = 0; j<dlines.size(); j++)
            {
            Vec4i l = dlines[j];
            double a1 = l[0];
            double b1 = l[1];
            double a2 = l[2];
            double b2 = l[3];
            

            int x = abs(a1-a2);
            int y = abs(b1-b2);

            if(a1>bias_x&&a1<x2-x1-bias_x&&a2>bias_x&&a2<x2-x1-bias_x){
                if(!x){
                    
                    
                    line(crop, Point(a1, b1), Point(a2, b2), Scalar(0, 255, 0), 1, CV_AA);
                    vector<double>keypoint ;
                    if(b1<b2){
                        keypoint.push_back(b1);
                        keypoint.push_back(b2);
                        list_keypoint.push_back(keypoint);
                        keypoint.clear();
                    }
                    else{
                        keypoint.push_back(b2);
                        keypoint.push_back(b1);
                        list_keypoint.push_back(keypoint);
                        keypoint.clear();
                    }
                    
                }
                else if(y/x>30){
                     line(crop, Point(a1, b1), Point(a2, b2), Scalar(0, 255, 0), 1, CV_AA);
                    // if(b1<b2){
                    //     list_keypoint.push_back(vector<double>{b1, b2});
                    // }
                    // else{
                    //     list_keypoint.push_back(vector<double>{b2, b1});
                    // }
                    vector<double>keypoint ;
                    if(b1<b2){
                        keypoint.push_back(b1);
                        keypoint.push_back(b2);
                        list_keypoint.push_back(keypoint);
                        keypoint.clear();
                    }
                    else{
                        keypoint.push_back(b2);
                        keypoint.push_back(b1);
                        list_keypoint.push_back(keypoint);
                        keypoint.clear();
                    }
                }
            }          
        }
        // lsd->drawSegments(img, dlines); //lines为上步直线检测的结果
        // cout << dlines.size() <<endl;
        double ratio = merge_ratio(0, y2-y1, list_keypoint);
        cout<< ratio <<endl;
        // putText(crop, ratio, (10, 400), FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        if(ratio<=0.995){
            num_unnormal++;

        }
  
        images_lsd.push_back(crop);


            }
        }
        
    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "It took " << duration_ms/num_total << " ms. for one picture" << std::endl;
    cout<<"total picture number: "<< num_total<<endl;
    cout<<"unnormal picture number: " << num_unnormal << endl;
    cout<<"unnormal rate: "<<num_unnormal/double(num_total)<<endl;


    return images_lsd;
}

int main(int argc, char** argv){

    CommandLineParser parser(argc, argv,
                                "{in_dir   i|/home/zhaoyx/platform_data/test/1_normal/*.jpg|input image}"
                                "{out_dir   o|/home/zhaoyx/platform_data/test/1_normal_out/|output dictionary}"
                                "{x1   x1|0|}"
                                "{x2   x2|1280|}"
                                "{y1   y1|0|}"
                                "{y2   y2|960|}"
                                "{bias_x   bias_x|0|}"                               
                                "{help    h|false|show help message}");
    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }
    parser.printMessage();
    String in_dir = parser.get<String>("in_dir");
    String out_dir = parser.get<String>("out_dir");
    int x1  = parser.get<int>("x1");
    int x2  = parser.get<int>("x2");
    int y1  = parser.get<int>("y1");
    int y2  = parser.get<int>("y2");
    int bias_x  = parser.get<int>("bias_x");

    //遍历得到目标文件中所有的.jpg文件
    vector<Mat> images = read_images_in_folder(in_dir);
    vector<Mat> imgaes_lsd = lsd_detector_unnormal(images, x1, x2, y1, y2, bias_x);
    write_images_to_folder(imgaes_lsd, out_dir);
    
    return 0;
}

//./test -x1=570 -x2=620 -y1=80 -y2=720 -bias_x=10
