# -*- coding: utf-8 -*-
# !/usr/bin/env python
import time
import hashlib


# ========================
# 时间相关
# ========================
# 函数耗时修饰器
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始时间
        result = func(*args, **kwargs)  # 执行原始函数
        end_time = time.time()  # 记录函数结束时间
        elapsed_time = end_time - start_time  # 计算函数执行时间
        print(f"函数 {func.__name__} 执行时间: {elapsed_time:.2f} 秒")
        return result
    return wrapper

@timing
def test():
    time.sleep(1)
    

# ========================
# 进度相关
# ========================
# 刷新打印
def flush_print(msg):
    print(msg, end='\r')

# 进度条
def flush_progress(total, current, msg='进度'):
    block = '█'
    blank = ' '
    percent = current / total
    percent = round(percent, 2)
    percent = percent * 100
    percent = int(percent)
    s = f'{msg}: {percent}% |'
    for i in range(50):
        if i < percent / 2:
            s += block
        else:
            s += blank
    s += '|'
    flush_print(s)
    
def test_flush_print():
    total = 100
    try:
        for i in range(total):
            msg = f'进度: {i}/{total}'
            flush_print(msg)
            time.sleep(0.1)
        flush_print('进度: 完成')
    except KeyboardInterrupt:
        flush_print('进度: 中断')

def test_flush_progress():
    total = 20
    try:
        for i in range(total):
            flush_progress(total, i)
            time.sleep(0.1)
        flush_progress(total, total)
        print()
        print('进度: 完成')
    except KeyboardInterrupt:
        flush_print('进度: 中断')


# ========================
# md5相关
# ========================
# 获取文件的md5
def get_md5(file_path):
    f = open(file_path, 'rb')
    md5_obj = hashlib.md5()
    md5_obj.update(f.read())
    hash_code = md5_obj.hexdigest()
    f.close()
    return hash_code

def test_get_md5():
    md5 = get_md5('data/video/fjdc.mp4')
    print(md5)
    
if __name__ == '__main__':
    test_flush_progress()