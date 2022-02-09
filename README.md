
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



## C++版本：    

1. 进入到lsd_c++目录    
2. mkdir build
3. cd build    
3. cmake ..    
4. make    
5. ./test -x1=570 -x2=620 -y1=80 -y2=720 -bias_x=10



