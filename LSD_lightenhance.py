# coding=utf-8
import cv2
import numpy as np
import time
import os

import argparse
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance



def compute(img, min_percentile, max_percentile):
	# """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    # """图像亮度增强"""
    if get_lightness(src)>1000:
        print("图片亮度足够，不做增强")
        return src
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    else:
        max_percentile_pixel, min_percentile_pixel = compute(src, 1,99)

        # 去掉分位值区间之外的值
        src[src>=max_percentile_pixel] = max_percentile_pixel
        src[src<=min_percentile_pixel] = min_percentile_pixel

        # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
        out = np.zeros(src.shape, src.dtype)
        cv2.normalize(src, out, 255*0.1, 255*0.9,cv2.NORM_MINMAX)

        return out

def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:,:,2].mean()
    # print("亮度"+str(lightness))

    return  lightness

# 对比度增强
def contrastEnhance(src):
    #对比度增强
    src = Image.fromarray(src)
    src = src.convert("RGB")
    enh_con = ImageEnhance.Contrast(src)  
    contrast = 2
    src = enh_con.enhance(contrast)
    out = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR) # PIL转cv2

    return out


# 纵坐标区域覆盖率
def merge_ratio(ymin,ymax, intervals):
    
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # 如果列表为空，或者当前区间与上一区间不重合，直接添加
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 否则的话，我们就可以与上一区间进行合并
            merged[-1][1] = max(merged[-1][1], interval[1])
        

    # 范围检测
    for line in merged:
        if line[0] >= ymin:
            break
        elif line[1] > ymin:
            line[0] = ymin
            break
        else:
            merged.remove(line)
    for line in merged[::-1]:
        if line[1] <= ymax:
            break
        elif line[0] < ymax:
            line[1] = ymax
            break
        else:
            merged.remove(line)

    # print(merged)

    # 计算长度
    length = 0
    for line in merged:
        # print(length)
        length += line[1]-line[0]
            
    return length/(ymax-ymin)



# 图像检测主函数，检测异常率
# TODO：
# 函数接口化，文件路径等方便调用
# 计算准确率：目前基于阈值计算
# 图片类型标注方法，例如在normal文件夹里，ratio就应该大于0.99
def lsd_detector_unnormal(pic_dir,pic_out):


    # 读取输入图片
    directory_name = pic_dir
    # directory_name = ("unnormal_pic/")
    start = time.clock()

    num_total = 0
    num_unnormal = 0
    num_light = 0

    for fileName in os.listdir(directory_name):
        num_total += 1

        img0 = cv2.imread(directory_name + '/' + fileName)
        cv2.circle(img0, (args.x1+args.bias_x, args.y1), 1, (255, 0, 0), 4)
        cv2.circle(img0, (args.x1+args.bias_x, args.y2), 1, (255, 0, 0), 4)
        cv2.circle(img0, (args.x2-args.bias_x, args.y2), 1, (255, 0, 0), 4)
        cv2.circle(img0, (args.x2-args.bias_x, args.y1), 1, (255, 0, 0), 4)

        # #对比度增强
        # img0 = contrastEnhance(img0)
        # img0 = aug(img0)


        #图像亮度增强
        # print(fileName)
        crop = img0[args.y1:args.y2, args.x1:args.x2]
        num_light += get_lightness(crop)


        if get_lightness(crop)<10:
            crop = aug(crop)


        if get_lightness(crop)>50:
            crop = aug(crop)
            crop = contrastEnhance(crop)
        # if get_lightness(crop)>100:
        #     crop = aug(crop)
        #     crop = contrastEnhance(crop)
        

        # crop = aug(crop)
        

        # gamma 变换 一般在亮度增强之后
#         img0 = np.power(img0/float(np.max(img0)), 1.2)*255
#         img0 = img0.astype(np.float32)
        # crop = np.power(crop/float(np.max(crop)), 1.2)*255
        # crop = crop.astype(np.float32)


        # crop=crop.astype(np.uint8)
        cv2.imwrite('2.jpg', crop)
        

        # 将彩色图片转换为灰度图片
        img = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
       
        # 创建一个LSD对象
        lsd = cv2.createLineSegmentDetector(0)
        # 执行检测结果
        
        dlines = lsd.detect(img)


         

        if dlines[0] is not None:
            j = 0
            list_keypoint = []
            for dline in dlines[0]:
                # print(dline)

                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                # print(x0, y0, x1, y1)
                x = abs(x1-x0)
                y = abs(y1-y0)

                if x1>0+args.bias_x and x1<args.x2-args.x1-args.bias_x and x0>0+args.bias_x and x0 <args.x2-args.x1 :
                    # if x!=0:
                    #     print(y/x)

                    if x==0:
                        j+=1
                        
                        # cv2.line(img0, (args.x1+x0, args.y1+y0), (args.x1+x1,args.y1+y1), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(crop, (x0, y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)
                        if y0<y1:
                            list_keypoint.append([y0,y1])
                        else:
                            list_keypoint.append([y1,y0])
                    elif y/x>args.tan:
                        j+=1
                        
                        # cv2.line(img0, (args.x1+x0, args.y1+y0), (args.x1+x1,args.y1+y1), (0,255,0), 1, cv2.LINE_AA)
                        cv2.line(crop, (x0, y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)
                        if y0<y1:
                            list_keypoint.append([y0,y1])
                        else:
                            list_keypoint.append([y1,y0])

            # cv2.putText(img0, ratio, (600, 400), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            # ratio = str(merge_ratio(63, 736, list_keypoint))
            ratio = merge_ratio(0, args.y2-args.y1, list_keypoint)
            cv2.putText(img0, str(ratio), (600, 400), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            # print(ratio)
            if(ratio <= 0.995):
                num_unnormal += 1
                print("异常"+fileName)
                if args.write_unnor:
                    cv2.imwrite(pic_out + '/'+ fileName, crop)
                
            else:
                # print("正常"+fileName)
                if args.write_nor:
                    cv2.imwrite(pic_out + '/'+ fileName, crop)

            # 显示并保存结果
            
            # cv2.imwrite('unnormal_pic_out/'+ fileName, img0)
        else:
            if args.write_unnor:
                cv2.imwrite(pic_out + '/'+ fileName, crop)
            num_unnormal += 1
        
    time_detect = (time.clock() - start)
    print('***************处理速度***************')
    print('图片处理总时长为:',time_detect,'s')
    print('平均每张图片处理用时为（异常判断+处理后图片存储）:',time_detect/num_total,'s')
    print('***************检测结果***************')
    print('图片检测数量为:',num_total,'张')
    print('检测到异常图片:',num_unnormal,'张')
    print('该目录下图片异常率为:',num_unnormal/num_total)
    print("平均亮度：", num_light/num_total)

    return 




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help='./example')
    parser.add_argument('--out_dir', type=str, help='./')
    parser.add_argument('--x1', type=int, help='left x', default = 0)
    parser.add_argument('--x2', type=int, help='right x', default = 1280)
    parser.add_argument('--y1', type=int, help='up y', default = 0)
    parser.add_argument('--y2', type=int, help='down y', default = 960)
    parser.add_argument('--bias_x', type=int, help='bias between edges and line to detect line', default = 0)
    # parser.add_argument('--bias_y', type=int, help='bias between edges and line to caculate result', default = 0)
    parser.add_argument('--tan', type = int, help = 'detect tan', default = 30)
    parser.add_argument('--write_nor', action = 'store_true' , help = 'store normal picture', default = False)
    parser.add_argument('--write_unnor', action = 'store_true' , help = 'store unnormal picture', default = False)

    args = parser.parse_args()
    if os.path.exists(args.out_dir) == False:
        os.makedirs(args.out_dir)
    lsd_detector_unnormal(args.in_dir, args.out_dir)

