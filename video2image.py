"""视频分帧"""
import cv2
 
vc = cv2.VideoCapture("/mnt/nfs/10.26.data/2021-10-26-0422_9-CapObj.AVI")
n = 1  # 计数
 
if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False
 
timeF = 10  # 视频帧计数间隔频率
 
i = 0
while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    if (n % timeF == 1):  # 每隔timeF帧进行存储操作

        i += 1
        print(i)
        # if i>1000:
        cv2.imwrite("/home/zhaoyx/platform_data/10.26/normal/0422_"+str(i)+'.jpg', frame)  # 存储为图像
        print("done")
    n = n + 1
    cv2.waitKey(1)
vc.release()

