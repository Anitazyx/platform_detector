
# **屏蔽门检测算法Python与C++版本**
###### 核心代码分别是python LSD_lightenhance.py与lsd_c++/platform_detector.cpp
### Python版本：    

1. pip install -r requirements.txt    
2. python video2image.py 视频流分帧    
3. python LSD_lightenhance.py         

    --in_dir {输入图像路径}         
    --out_dir {输出图像路径}         
    --x1 {crop左上x}        
    --x2 {crop右下x}        
    --y1 {crop左上y}        
    --y2 {crop右下y，将直线边缘裁剪到crop框的边缘}        
    --bias_x {直线检测只在（x1+bia_x）到（x2-x1-bias_x）范围内进行}         
    --tan {只检测斜率大于tan的直线，default=30}        
    --write_unnor {写入检测异常图片}        
    --write_nor {写入检测正常图片}  

4. python pic2video.py 图片转成视频流

>Python测试结果记录在：[实验结果](http://confluence.jxresearch.com:8090/pages/viewpage.action?pageId=82874179)

## C++版本：    

1. 进入到lsd_c++目录    
2. mkdir build
3. cd build    
3. cmake ..    
4. make    
5. ./test -x1=570 -x2=620 -y1=80 -y2=720 -bias_x=10
#### 运行结果
1. 环境使用tx2 192.168.170.148
2. opencvdir为/home/nvidia/opencv-3.4.9/build
3. 从读取图片到存储图片的处理流程，一张图片耗时30-40ms

## 测试数据与参数设置
文件夹：/home/zhaoyx/platform_data/    
###### 设置参数        
    10.19交大三段视频（路径：/home/zhaoyx/platform_data/test）：            
        --x1 570 --x2 620 --y1 80 --y2 720（第一段第二段）            
        --x1 570 --x2 620 --y1 80 --y2 730（第三段）        
    10.26交大亮度强视频（路径：/home/zhaoyx/platform_data/10.26）：           
        --x1 570 --x2 625 --y1 140 --y2 805 --bias_x 15（0414）           
        --x1 575 --x2 630 --y1 135 --y2 800 --bias_x 15（0418）           
        --x1 575 --x2 630 --y1 135 --y2 800 --bias_x 15（0420）            
        --x1 575 --x2 630 --y1 135 --y2 800 --bias_x 15（0421）            
        --x1 575 --x2 630 --y1 135 --y2 800 --bias_x 15（0422）            
        --x1 560 --x2 620 --y1 140 --y2 810 --bias_x 15（0423）
