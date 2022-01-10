# jpg2video.py
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import argparse

def write(images, outimg=None, fps=5, size=None, is_color=True, format="XVID", outvid='demo.avi'):
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if image.split('.')[-1] != 'jpg':
            continue
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def jpg2video(in_dir, out_dir, fps, out_size):
    filenames = os.listdir(in_dir)
    filenames = sorted(filenames)
    filenamex = [os.path.join(in_dir, x) for x in filenames]
    video_name = in_dir.split('/')[-1] + '.avi'
    write(filenamex, fps=fps, size=out_size, outvid=os.path.join(out_dir, video_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help='./example')
    parser.add_argument('--out_dir', type=str, help='./')
    parser.add_argument('--fps', type=int, help='25')
    parser.add_argument('--out_height', type=int, help='640')
    parser.add_argument('--out_width', type=int, help='480')
    args = parser.parse_args()
    jpg2video(args.in_dir, args.out_dir, args.fps, (args.out_width, args.out_height))