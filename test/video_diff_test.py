# -*- coding: utf-8 -*-
# !/usr/bin/env python
import sys
sys.path.append("../") # 插入相对路径, 最好是项目根路径
import cv2
import utils.Utils as Utils    
from video_diff import *


# 多视频比较测试
@Utils.timing
def test_multi_video_compare():
    folder = '../data/video/'
    mv = MultiVideoHandler(folder)
    print(mv.sim_set)

# 删除重复视频测试
@Utils.timing
def test_delete_duplicate():
    folder = '../data/video/'
    mv = MultiVideoHandler(folder, frame_ratio=0.05)
    mv.delete_duplicate()

# 视频基本信息测试
@Utils.timing
def test_video_info():
    v = VideoHandler('../data/video/1.mp4', frame_ratio=0.01)
    info = v.get_video_info()
    print(info)

# 图片相似度计算测试
@Utils.timing
def test_image_diff():
    image1 = Image.open('../data/img/1.jpg')
    image2 = Image.open('../data/img/2.jpg')
    cosin = ImageHandler.image_similarity_vectors_via_numpy(image1, image2)
    print('图片余弦相似度', cosin)


if __name__ == '__main__':
    test_image_diff()
    test_video_info()
    test_multi_video_compare()
    test_delete_duplicate()