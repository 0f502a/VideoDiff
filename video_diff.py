# -*- coding: utf-8 -*-
# !/usr/bin/env python
import os
import cv2
import uuid
import Utils
from PIL import Image
from tqdm import tqdm
from numpy import average, dot, linalg


# ========================
# 视频处理类
# ========================
class VideoHandler(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_name = video_path.split('/')[-1].split('.')[0]
        self.video_info = dict()
        self.cap = None
        self.tmp_files = []
        self.load()
    
    # 析构函数：退出时释放资源、清除临时文件
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print('释放视频资源: ', self.video_path)
        if len(self.tmp_files) > 0:
            self.__clear_tmp(self.tmp_files)
            # print('清除临时文件: ', self.tmp_files)

    def load(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        self.cap = cap
        self.video_info = self.get_video_info()

    # 视频信息
    def get_video_info(self) -> dict:
        if self.video_info is not None and len(self.video_info) > 0:
            return self.video_info
        cap = self.cap
        if cap is None:
            return {}
        # 帧率
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        # 分辨率-宽度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 分辨率-高度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 总帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 时长，单位s
        duration = frame_count / fps
        # md5
        md5 = Utils.get_md5(self.video_path)
        
        return {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'duration': duration,
            'md5': md5
        }

    # 视频抽帧
    def save_img(self, save_path='tmp/', ratio=0.15) -> list:
        cap = self.cap
        if cap is None:
            return []
        video_info = self.get_video_info()
        timeF=15 # 视频帧计数间隔频率
        if 0 < ratio < 1:
            timeF = int(video_info["frame_count"] * ratio) # 按全局帧数比例抽帧
        frame_ids = [i for i in range(0, video_info["frame_count"], timeF)] # 设置要抽取的帧id
        images = [] # 保存的图片地址
        print('开始抽帧:', self.video_path)
        for fid in tqdm(frame_ids, desc='抽帧进度'):
            rval, frame = cap.read()
            if not rval:
                break
            video_name = self.video_name+'_'+str(uuid.uuid1())
            img_full_name = save_path + video_name + '_' + str(fid) + '.jpg'
            cv2.imwrite(img_full_name, frame) # 存储为图像
            images.append(img_full_name)
            cv2.waitKey(1)
            # 跳帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        print('抽帧完成:', self.video_path)
        self.tmp_files.extend(images)
        return images
    
    # 清除临时文件
    def __clear_tmp(self, tmp_files):
        for f in tmp_files:
            os.remove(f)

    # 抽帧比对
    @staticmethod
    def compare_by_frame(v1, v2, frame_ratio=0.15) -> float:
        v1_imgs = v1.save_img(ratio=frame_ratio)
        v2_imgs = v2.save_img(ratio=frame_ratio)
        if len(v1_imgs) != len(v2_imgs):
            return 0.0
        avg_cosin = 0
        for i in range(len(v1_imgs)):
            img1 = Image.open(v1_imgs[i])
            img2 = Image.open(v2_imgs[i])
            cosin = ImageHandler.image_similarity_vectors_via_numpy(img1, img2)
            avg_cosin += cosin
        avg_cosin /= len(v1_imgs)
        print('视频相似度(cos):', avg_cosin)
        return avg_cosin

    # 判断两个视频是否相同
    @staticmethod
    @Utils.timing
    def is_same(v1_path, v2_path, threshold=0.9, frame_ratio=0.15):
        v1 = VideoHandler(v1_path)
        v2 = VideoHandler(v2_path)
        if v1.cap is None or v2.cap is None:
            print('视频加载失败')
            return False
        v1_info = v1.video_info
        v2_info = v2.video_info
        print('视频基本信息:')
        print("v1: ", v1_info)
        print("v2: ", v2_info)
        # 条件1: md5相同
        if v1_info["md5"] == v2_info["md5"]:
            return True
        # 条件2: md5不一致情况下，判断视频基本信息是否相同
        if v1_info["fps"] != v2_info["fps"]:
            return False
        if v1_info["frame_count"] != v2_info["frame_count"]:
            return False
        # 条件3: 抽帧相似度高于阈值（考虑视频相同但质量不一致的情况）
        cosin = VideoHandler.compare_by_frame(v1, v2, frame_ratio)
        if cosin < threshold:
            return False
        return True


def test_video_info():
    video_path = 'data/video/fjdc.mp4'
    info = VideoHandler.get_video_info(video_path)
    print(info)


# 视频跳帧测试
def test_video_skip_frame():
    v = VideoHandler('data/video/fjdc.mp4')
    info = v.get_video_info()
    frame_ids = [i for i in range(1, info["frame_count"], 15)]
    for fid in frame_ids:
        rval, frame = v.cap.read()
        if not rval:
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        v.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)


# ========================
# 图片处理类
# ========================
class ImageHandler(object):
    def __init__(self):
        pass

    # 对图片进行统一化处理
    @staticmethod
    def get_thum(image, size=(64, 64), greyscale=False):
        # 利用image对图像大小重新设置, Image.LANCZOS为高质量的
        image = image.resize(size, Image.LANCZOS)
        if greyscale:
            # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
            image = image.convert('L')
        return image

    # 计算图片的余弦距离
    @staticmethod
    def image_similarity_vectors_via_numpy(image1, image2):
        image1 = ImageHandler.get_thum(image1)
        image2 = ImageHandler.get_thum(image2)
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            # linalg=linear（线性）+algebra（代数），norm则表示范数
            # 求图片的范数
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # dot返回的是点积，对二维数组（矩阵）进行计算
        res = dot(a / a_norm, b / b_norm)
        return res


def test_image_diff():
    image1 = Image.open('data/img/1.jpeg')
    image2 = Image.open('data/img/2.jpeg')
    cosin = ImageHandler.image_similarity_vectors_via_numpy(image1, image2)
    print('图片余弦相似度', cosin)


# ========================
# main
# ========================
def main():
    v1 = 'data/video/fjdc.mp4'
    v2 = 'data/video/fjdc2.mp4'
    res = VideoHandler.is_same(v1, v2, frame_ratio=0.05)
    if res:
        print('视频相似')
    else:
        print('视频不相似')


if __name__ == '__main__':
    main()