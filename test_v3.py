import sys
sys.path.insert(0, r'D:\Data\Playground\test\20260429')
from color_transfer_gui import color_transfer, process_one_pair
from pathlib import Path
import cv2
import time

# 1. 测试修正后的算法效果
src = cv2.imread(r'D:\Data\Playground\test\20260429\测试数据集\图像.jpg')
tar = cv2.imread(r'D:\Data\Playground\test\20260429\测试数据集\图像2.jpg')

print("Testing corrected algorithm...")
result = color_transfer(src, tar)
out = r'D:\Data\Playground\test\20260429\test_result_v3.jpg'
cv2.imwrite(out, result)
print(f"Saved corrected result to {out}")

# 2. 测试单张性能（多次取平均）
times = []
for _ in range(10):
    t0 = time.perf_counter()
    _ = color_transfer(src, tar)
    t1 = time.perf_counter()
    times.append(t1 - t0)
print(f"Single image avg time: {sum(times)/len(times)*1000:.2f} ms")

# 3. 测试多进程函数可调用性
print("Testing process_one_pair...")
code, msg = process_one_pair(
    "图像",
    r'D:\Data\Playground\test\20260429\测试数据集\图像.jpg',
    r'D:\Data\Playground\test\20260429\测试数据集\图像2.jpg',
    r'D:\Data\Playground\test\20260429'
)
print(f"Result: code={code}, msg={msg}")
