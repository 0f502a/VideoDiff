# -*- coding: utf-8 -*-
# !/usr/bin/env python
import os
import cv2
import time
import uuid
import utils.Utils as Utils
from PIL import Image
from tqdm import tqdm
from numpy import average, dot, linalg


# ========================
# 多视频处理类
# ========================
class MultiVideoHandler(object):
    def __init__(self, folder, frame_ratio=0.15) -> None:
        self.folder = folder    # 视频文件夹
        self.videos = []        # 视频列表
        self.sim_set = None     # 相似视频的组合
        self.frame_ratio = frame_ratio
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
        print(">>> 视频数量: {} <<<".format(len(video_paths)))
        for p in tqdm(video_paths, desc='视频加载'):
            v = VideoHandler(p, frame_ratio=self.frame_ratio)
            if v and v.cap:
                self.videos.append(v)
        print(">>> 成功加载视频数：{} <<<".format(len(self.videos)))
        # 计算视频间的相似性
        self.sim_set, cnt = self.compare()
        print('>>> 共有{}组相似视频, 共{}个相似视频 <<<'.format(len(self.sim_set), cnt)) 

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
        for i in range(len(sim_videos)):
            if len(sim_videos[i]) > 0:
                vs = [i] + sim_videos[i]
                vs.sort()
                st.add(tuple(vs))
        cnt = 0
        for tup in st:
            cnt += len(tup)
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
                v.release(v.cap) # 释放视频资源
                os.remove(v.video_path)
            del_cnt += len(v_lst) - 1
        print('>>> 删除视频数量 {} <<<'.format(del_cnt))


# ========================
# 视频处理类
# ========================
class VideoHandler(object):
    def __init__(self, video_path, frame_ratio=0.15):
        # 初始化参数
        self.video_path = video_path
        self.frame_ratio = frame_ratio
        self.video_name = video_path.split('/')[-1].split('.')[0]
        self.__video_info = dict()
        self.__frames = list()
        # 载入视频信息、预处理
        self.cap = self.load(video_path)
        if self.cap is None:
            return None
        self.__video_info = self.extract_video_info(cap=self.cap, video_path=video_path)
        self.__frames = self.save_img(cap=self.cap, video_path=video_path, ratio=frame_ratio)
    
    # 析构函数：退出时释放资源、清除临时文件
    def __del__(self):
        self.release(self.cap)
        self.__clear_tmp(self.__frames)
            
    # 加载视频
    @staticmethod
    def load(video_path=""):
        if not os.path.exists(video_path):
            print("[load] video_path not exists")
            return None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("[load] cap.isOpened()=False")
                return None
            return cap
        except Exception as e:
            print(e)
            return None

    # 释放视频资源
    @staticmethod
    def release(cap=None):
        if cap and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            # print('释放视频资源: ', self.video_path)
    
    # 视频信息
    def get_video_info(self) -> dict:
        return self.__video_info

    # 提取视频信息
    def extract_video_info(self, cap, video_path) -> dict:
        if cap is None or not cap.isOpened():
            print("[extract_video_info] read cap error")
            return {}
        if not os.path.exists(video_path):
            print(["[extract_video_info] read video_path error"])
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
        md5 = Utils.get_md5(video_path)
        # 获取创建时间
        file_info = os.stat(video_path)
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

    # 获取视频帧的列表
    def get_frames(self) -> list:
        return self.__frames

    # 视频抽帧
    def save_img(self, cap, video_path, save_path='tmp/', ratio=0.15) -> list:
        # 读取视频
        if cap is None or video_path == "":
            print("[save_img] cap or video_path is None")
            return []
        # 保存图片目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 设置视频帧计数间隔频率
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step=15 
        if 0 < ratio < 1:
            frame_step = round(1 / ratio)
        frame_ids = [i for i in range(0, frame_count, frame_step)] # 设置要抽取的帧id
        # 图片列表
        images = []
        video_name = video_path.split('/')[-1].split('.')[0]
        for fid in frame_ids:
            rval, frame = cap.read()
            if not rval:
                break
            # 图片名称: {文件夹}{视频名}_{随机uuid}_{帧id}.jpg
            img_full_name = "{}{}_{}_{}.{}".format(
                save_path,
                video_name,
                str(uuid.uuid1()),
                fid,
                'jpg'
            )
            cv2.imwrite(img_full_name, frame) # 存储为图像
            images.append(img_full_name)
            cv2.waitKey(1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid) # 跳帧
        return images
    
    # 清除临时文件
    def __clear_tmp(self, tmp_files):
        for f in tmp_files:
            os.remove(f)

    # 抽帧比对图片的余弦相似度
    @staticmethod
    def cal_cos_similarity(v1, v2) -> float:
        v1_imgs = v1.get_frames()
        v2_imgs = v2.get_frames()
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
        v1_info = v1.get_video_info()
        v2_info = v2.get_video_info()
        if v1_info == {} or v2_info == {}:
            print('[is_same] 视频信息为空')
            return False
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


def main():
    v = VideoHandler('data/video/1.mp4', frame_ratio=0.01)
    info = v.get_video_info()
    print(info)

if __name__ == '__main__':
    main()