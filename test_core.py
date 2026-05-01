import sys
sys.path.insert(0, r'D:\Data\Playground\test\20260429')
from color_transfer_gui import color_transfer
import cv2

src = cv2.imread(r'D:\Data\Playground\test\20260429\测试数据集\图像.jpg')
tar = cv2.imread(r'D:\Data\Playground\test\20260429\测试数据集\图像2.jpg')

if src is None or tar is None:
    print('Failed to load test images')
    sys.exit(1)

print(f'Source size: {src.shape}, Target size: {tar.shape}')
result = color_transfer(src, tar)
print(f'Result size: {result.shape}')

out_path = r'D:\Data\Playground\test\20260429\test_result.jpg'
ok = cv2.imwrite(out_path, result)
print(f'Saved to {out_path}: {ok}')
