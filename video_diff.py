# -*- coding: utf-8 -*-
# !/usr/bin/env python
import os
import cv2
import time
import uuid
import Utils
from PIL import Image
from tqdm import tqdm
from numpy import average, dot, linalg


# ========================
# 多视频处理类
# ========================
class MultiVideoHandler(object):
    def __init__(self, folder) -> None:
        self.folder = folder    # 视频文件夹
        self.videos = []        # 视频列表
        self.sim_set = None     # 相似视频的组合
        self.load()

    def __del__(self):
        for v in self.videos:
            del v # 释放视频资源
    
    # 加载文件夹下所有视频
    # 预处理：1. 计算视频基本信息、抽帧；2. 对比文件夹下所有视频之间的相似性
    def load(self):
        if not os.path.exists(self.folder):
            return
        files = os.listdir(self.folder)
        video_paths = [os.path.join(self.folder, f) for f in files if f.endswith('.mp4')]
        for p in tqdm(video_paths, desc='视频加载'):
            v = VideoHandler(p)
            self.videos.append(v)
        # 计算视频间的相似性
        self.sim_set, cnt = self.compare()

    # 比较文件夹下所有视频之间的相似性，返回相似视频的组合、相似视频的数量
    def compare(self, threshold=0.99):
        sim_videos = [[] for i in range(len(self.videos))]
        # TODO: 优化比对算法, 降低时间复杂度
        for i in tqdm(range(len(self.videos)), desc='视频比对'):
            for j in range(i+1, len(self.videos)):
                v1 = self.videos[i]
                v2 = self.videos[j]
                res = VideoHandler.is_same(v1, v2, threshold=threshold)
                if res:
                    sim_videos[i].append(j)
                    sim_videos[j].append(i)
        st = set()
        cnt = 0
        for i in range(len(sim_videos)):
            if len(sim_videos[i]) > 0:
                vs = [i] + sim_videos[i]
                vs.sort()
                st.add(tuple(vs))
                cnt += len(vs)
        return st, cnt # 返回相似视频的组合、相似视频的数量
    
    # 删除重复视频
    # 优先级：1.分辨率width 从大到小；2.修改时间modify_time 从新到旧
    def delete_duplicate(self):
        if self.sim_set is None or len(self.sim_set) == 0:
            print('没有重复视频')
            return
        del_cnt = 0
        for tup in self.sim_set:
            v_lst = [self.videos[i] for i in tup]
            v_lst.sort(key=lambda x: (x.video_info['width'], x.video_info['modify_time']), reverse=True)
            for v in v_lst[1:]:
                print('删除视频', v.video_path)
                os.remove(v.video_path)
            del_cnt += len(v_lst) - 1
        print('删除视频数量', del_cnt)

@Utils.timing
def test_multi_video_compare():
    folder = 'data/xiaoe/'
    mv = MultiVideoHandler(folder)
    print(mv.sim_set)

@Utils.timing
def test_delete_duplicate():
    folder = 'data/xiaoe2/'
    mv = MultiVideoHandler(folder)
    mv.delete_duplicate()


# ========================
# 视频处理类
# ========================
class VideoHandler(object):
    def __init__(self, video_path, frame_ratio=0.15):
        self.video_path = video_path
        self.video_name = video_path.split('/')[-1].split('.')[0]
        self.video_info = dict()
        self.cap = None
        self.frames = []
        self.frame_ratio = frame_ratio
        self.load()
    
    # 析构函数：退出时释放资源、清除临时文件
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            # print('释放视频资源: ', self.video_path)
        if len(self.frames) > 0:
            self.__clear_tmp(self.frames)
            # print('清除临时文件: ', self.frames)

    def load(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        self.cap = cap
        self.video_info = self.get_video_info()
        self.frames = self.save_img(ratio=self.frame_ratio)
        
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
        # 获取创建时间
        file_info = os.stat(self.video_path)
        create_time = time.localtime(file_info.st_ctime)
        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S", create_time)

        # 获取修改时间
        modify_time = time.localtime(file_info.st_mtime)
        modify_time_str = time.strftime("%Y-%m-%d %H:%M:%S", modify_time)
        
        return {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'duration': duration,
            'md5': md5,
            'create_time': create_time_str,
            'modify_time': modify_time_str
        }

    # 视频抽帧
    def save_img(self, save_path='tmp/', ratio=0.15) -> list:
        cap = self.cap
        if cap is None:
            return []
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        video_info = self.get_video_info()
        timeF=15 # 视频帧计数间隔频率
        if 0 < ratio < 1:
            timeF = int(video_info["frame_count"] * ratio) # 按全局帧数比例抽帧
        frame_ids = [i for i in range(0, video_info["frame_count"], timeF)] # 设置要抽取的帧id
        images = [] # 保存的图片地址
        for fid in frame_ids:
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
        return images
    
    # 清除临时文件
    def __clear_tmp(self, tmp_files):
        for f in tmp_files:
            os.remove(f)

    # 抽帧比对图片的余弦相似度
    @staticmethod
    def cal_cos_similarity(v1, v2) -> float:
        v1_imgs = v1.frames
        v2_imgs = v2.frames
        if len(v1_imgs) != len(v2_imgs) or len(v1_imgs) == 0 or len(v2_imgs) == 0:
            return 0.0
        avg_cosin = 0
        for i in range(len(v1_imgs)):
            img1 = Image.open(v1_imgs[i])
            img2 = Image.open(v2_imgs[i])
            cosin = ImageHandler.image_similarity_vectors_via_numpy(img1, img2)
            avg_cosin += cosin
        avg_cosin /= len(v1_imgs)
        # print('视频相似度(cos):', avg_cosin)
        return avg_cosin

    # 判断两个视频是否相同
    @staticmethod
    def is_same(v1, v2, threshold=0.99):
        if v1.cap is None or v2.cap is None:
            print('视频加载失败')
            return False
        v1_info = v1.video_info
        v2_info = v2.video_info
        # 条件1: md5相同
        if v1_info["md5"] == v2_info["md5"]:
            return True
        # 条件2: md5不一致情况下，判断视频基本信息是否相同
        if v1_info["fps"] != v2_info["fps"]:
            return False
        if v1_info["frame_count"] != v2_info["frame_count"]:
            return False
        # 条件3: 抽帧相似度高于阈值（考虑视频相同但质量不一致的情况）
        cosin = VideoHandler.cal_cos_similarity(v1, v2)
        if cosin < threshold:
            return False
        return True


def test_video_info():
    v = VideoHandler('data/video/1.mp4')
    info = v.get_video_info()
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
    v1 = VideoHandler('data/video/1.mp4')
    v2 = VideoHandler('data/video/2.mp4')
    res = VideoHandler.is_same(v1, v2)
    if res:
        print('视频相似')
    else:
        print('视频不相似')


if __name__ == '__main__':
    test_delete_duplicate()