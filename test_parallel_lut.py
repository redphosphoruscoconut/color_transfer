import sys
sys.path.insert(0, r'D:\Data\Playground\test\20260429')
from color_transfer_gui import process_one_pair
from concurrent.futures import ProcessPoolExecutor
import time
from pathlib import Path

if __name__ == '__main__':
    src_dir = Path(r'D:\Data\Playground\test\20260429\测试数据集')
    tar_dir = Path(r'D:\Data\Playground\test\20260429\测试数据集\bucket_output')
    out_dir = Path(r'D:\Data\Playground\test\20260429\test_output_lut')
    out_dir.mkdir(exist_ok=True)

    img_exts = {'.png', '.jpg', '.jpeg'}
    src_files = {f.stem: f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts}
    tar_files = {f.stem: f for f in tar_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts}

    pairs = [(stem, str(src_files[stem]), str(tar_files[stem]), str(out_dir), True, 33)
             for stem in sorted(src_files.keys()) if stem in tar_files]

    print(f"Found {len(pairs)} pairs, LUT export ON")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as exe:
        futs = [exe.submit(process_one_pair, *p) for p in pairs]
        results = [f.result() for f in futs]
    t1 = time.perf_counter()

    success = sum(1 for c, _ in results if c == 0)
    fail = len(results) - success
    print(f"Done: success={success}, fail={fail}, total={t1-t0:.2f}s")
    for c, m in results[:3]:
        print(f"  {m}")

    # 检查是否生成了cube
    cubes = list(out_dir.glob('*.cube'))
    print(f"Cube files generated: {len(cubes)}")
