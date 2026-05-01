import sys
sys.path.insert(0, r'D:\Data\Playground\test\20260429')
from color_transfer_gui import color_transfer, numpy_to_pixmap
import cv2
from pathlib import Path

# 测试配对扫描逻辑
src_dir = Path(r'D:\Data\Playground\test\20260429\测试数据集')
tar_dir = Path(r'D:\Data\Playground\test\20260429\测试数据集\bucket_output')
img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

src_files = {f.stem: f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts}
tar_files = {f.stem: f for f in tar_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts}

pairs = []
for stem in sorted(src_files.keys()):
    if stem in tar_files:
        pairs.append((src_files[stem], tar_files[stem]))

print(f"Source: {len(src_files)}, Target: {len(tar_files)}, Pairs: {len(pairs)}")
for s, t in pairs[:5]:
    print(f"  {s.name} <-> {t.name}")

# 测试处理其中一对
if pairs:
    s, t = pairs[0]
    src_img = cv2.imread(str(s))
    tar_img = cv2.imread(str(t))
    result = color_transfer(src_img, tar_img)
    out = Path(r'D:\Data\Playground\test\20260429\test_result_v2.jpg')
    cv2.imwrite(str(out), result)
    print(f"Saved preview result to {out}")
    
    # 测试numpy_to_pixmap不崩溃
    pm = numpy_to_pixmap(result, max_size=400)
    print(f"QPixmap created: {pm is not None}, size: {pm.size() if pm else 'N/A'}")
