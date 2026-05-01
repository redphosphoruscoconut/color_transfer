import sys
import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QTextEdit, QMessageBox, QGroupBox, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage


# ==================== 数据模型 ====================

@dataclass
class PairItem:
    stem: str
    source_path: Path
    target_path: Path
    status: str = "待处理"


# ==================== 核心算法 ====================

def color_transfer(source_img: np.ndarray, target_img: np.ndarray) -> Optional[np.ndarray]:
    if source_img is None or target_img is None:
        return None

    h, w = source_img.shape[:2]
    if target_img.shape[0] != h or target_img.shape[1] != w:
        target_resized = cv2.resize(target_img, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        target_resized = target_img

    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = source_lab.copy()
    mean_src_l = np.mean(source_lab[:, :, 0])
    mean_tar_l = np.mean(target_lab[:, :, 0])
    result[:, :, 0] = source_lab[:, :, 0] - mean_src_l + mean_tar_l

    for i in range(1, 3):
        mean_src, std_src = np.mean(source_lab[:, :, i]), np.std(source_lab[:, :, i])
        mean_tar, std_tar = np.mean(target_lab[:, :, i]), np.std(target_lab[:, :, i])
        if std_src < 1e-6:
            std_src = 1e-6
        result[:, :, i] = (source_lab[:, :, i] - mean_src) * (std_tar / std_src) + mean_tar

    np.clip(result, 0, 255, out=result)
    return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)


def generate_cube_lut(source_path: str, target_path: str, output_path: str, grid_size: int = 33) -> bool:
    """
    基于 source/target 的 LAB 统计量生成 3D LUT (.cube)。
    变换逻辑与 color_transfer 完全一致：
      - L: 仅均值偏移（保留原图明暗分布）
      - A/B: 均值+标准差完整迁移
    """
    source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
    target_img = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if source_img is None or target_img is None:
        return False

    h, w = source_img.shape[:2]
    if target_img.shape[0] != h or target_img.shape[1] != w:
        target_resized = cv2.resize(target_img, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        target_resized = target_img

    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean_src_l = float(np.mean(source_lab[:, :, 0]))
    mean_tar_l = float(np.mean(target_lab[:, :, 0]))
    mean_src_a, std_src_a = float(np.mean(source_lab[:, :, 1])), float(np.std(source_lab[:, :, 1]))
    mean_tar_a, std_tar_a = float(np.mean(target_lab[:, :, 1])), float(np.std(target_lab[:, :, 1]))
    mean_src_b, std_src_b = float(np.mean(source_lab[:, :, 2])), float(np.std(source_lab[:, :, 2]))
    mean_tar_b, std_tar_b = float(np.mean(target_lab[:, :, 2])), float(np.std(target_lab[:, :, 2]))

    if std_src_a < 1e-6:
        std_src_a = 1e-6
    if std_src_b < 1e-6:
        std_src_b = 1e-6

    # 构建网格 BGR，值域 0-255
    steps = np.linspace(0, 255, grid_size, dtype=np.float32)
    b_grid, g_grid, r_grid = np.meshgrid(steps, steps, steps, indexing='ij')
    lut_img = np.stack([b_grid, g_grid, r_grid], axis=-1)  # (G,G,G,3) G=grid_size

    # 一次性转 LAB
    lut_lab = cv2.cvtColor(lut_img.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lut_lab = lut_lab.reshape(grid_size, grid_size, grid_size, 3)

    # 应用变换
    lut_lab[:, :, :, 0] = lut_lab[:, :, :, 0] - mean_src_l + mean_tar_l
    lut_lab[:, :, :, 1] = (lut_lab[:, :, :, 1] - mean_src_a) * (std_tar_a / std_src_a) + mean_tar_a
    lut_lab[:, :, :, 2] = (lut_lab[:, :, :, 2] - mean_src_b) * (std_tar_b / std_src_b) + mean_tar_b

    np.clip(lut_lab, 0, 255, out=lut_lab)

    # 转回 RGB（.cube 标准通常存 RGB）
    lut_rgb = cv2.cvtColor(lut_lab.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    lut_rgb = lut_rgb.reshape(grid_size, grid_size, grid_size, 3)
    lut_rgb /= 255.0

    # 写入 .cube：B 外层 -> G 中层 -> R 内层
    stem_name = Path(source_path).stem
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'TITLE "Color Transfer LUT - {stem_name}"\n')
        f.write(f'LUT_3D_SIZE {grid_size}\n\n')
        for b in range(grid_size):
            for g in range(grid_size):
                for r in range(grid_size):
                    val = lut_rgb[b, g, r]
                    f.write(f'{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}\n')
    return True


def process_one_pair(stem: str, source_path: str, target_path: str,
                     output_dir: str, export_lut: bool = False,
                     lut_size: int = 33) -> Tuple[int, str]:
    """
    独立进程执行的单任务函数。
    返回 (0=成功/1=失败, 消息)。
    """
    try:
        # ---- 图像处理 ----
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        target_img = cv2.imread(target_path, cv2.IMREAD_COLOR)

        if source_img is None:
            return (1, f"[失败] 无法读取源文件: {stem}")
        if target_img is None:
            return (1, f"[失败] 无法读取目标文件: {stem}")

        result = color_transfer(source_img, target_img)
        if result is None:
            return (1, f"[失败] 处理失败: {stem}")

        sp = Path(source_path)
        out_path = Path(output_dir) / f"{stem}_ct{sp.suffix}"
        ok = cv2.imwrite(str(out_path), result)
        if not ok:
            return (1, f"[失败] 保存失败: {stem}")

        msg = f"[完成] {stem}"

        # ---- LUT 导出 ----
        if export_lut:
            lut_path = Path(output_dir) / f"{stem}_ct.cube"
            lut_ok = generate_cube_lut(source_path, target_path, str(lut_path), lut_size)
            if lut_ok:
                msg += f"  (+LUT {lut_size}³)"
            else:
                msg += f"  (LUT生成失败)"

        return (0, msg)
    except Exception as e:
        return (1, f"[异常] {stem}: {e}")


def numpy_to_pixmap(img_bgr: np.ndarray, max_size: int = 600) -> Optional[QPixmap]:
    if img_bgr is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h
    bytes_per_line = 3 * new_w
    qimg = QImage(rgb.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ==================== 工作线程 ====================

class ProcessWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    pair_status = Signal(int, str)
    finished_signal = Signal(bool, str)

    def __init__(self, pairs: List[PairItem], output_dir: Path,
                 max_workers: int = 4, export_lut: bool = False, lut_size: int = 33):
        super().__init__()
        self.pairs = pairs
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.export_lut = export_lut
        self.lut_size = lut_size
        self._is_running = True
        self._executor: Optional[ProcessPoolExecutor] = None

    def stop(self):
        self._is_running = False
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                self._executor.shutdown(wait=False)

    def run(self):
        try:
            total = len(self.pairs)
            if total == 0:
                self.finished_signal.emit(False, "没有需要处理的配对。")
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"启动 {self.max_workers} 进程并行处理，共 {total} 对...")

            completed = 0
            failed = 0

            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            futures = {}

            for idx, pair in enumerate(self.pairs):
                if not self._is_running:
                    break
                self.pair_status.emit(idx, "排队中")
                future = self._executor.submit(
                    process_one_pair,
                    pair.stem,
                    str(pair.source_path),
                    str(pair.target_path),
                    str(self.output_dir),
                    self.export_lut,
                    self.lut_size
                )
                futures[future] = idx

            for future in as_completed(futures):
                if not self._is_running:
                    break
                idx = futures[future]
                try:
                    code, msg = future.result(timeout=300)
                except Exception as e:
                    code, msg = 1, f"[异常] 第{idx+1}对: {e}"

                status = "完成" if code == 0 else "失败"
                if code != 0:
                    failed += 1
                self.pair_status.emit(idx, status)
                self.log.emit(msg)
                completed += 1
                self.progress.emit(int(completed / total * 100))

            if self._is_running:
                self._executor.shutdown(wait=True)
                self.finished_signal.emit(
                    True,
                    f"处理结束：成功 {completed - failed} / 总计 {total}，输出目录：{self.output_dir}"
                )
            else:
                self.finished_signal.emit(False, "用户中断了处理。")

        except Exception as e:
            self.finished_signal.emit(False, f"调度异常: {e}")


# ==================== 主窗口 ====================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("色彩迁移批量处理工具 - LUT导出版")
        self.setMinimumSize(1100, 720)

        self.pairs: List[PairItem] = []
        self.worker: Optional[ProcessWorker] = None
        self._current_source_img: Optional[np.ndarray] = None
        self._current_target_img: Optional[np.ndarray] = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ===== 顶部路径区 =====
        path_group = QGroupBox("文件夹选择")
        path_layout = QVBoxLayout(path_group)

        def make_path_row(label_text, btn_text, slot):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            edit = QLineEdit()
            edit.setReadOnly(True)
            row.addWidget(edit)
            btn = QPushButton(btn_text)
            btn.clicked.connect(slot)
            row.addWidget(btn)
            return edit, row

        self.src_edit, r1 = make_path_row("Source (原图):", "浏览...", self.choose_source)
        self.tar_edit, r2 = make_path_row("Target (参考图):", "浏览...", self.choose_target)
        self.out_edit, r3 = make_path_row("Output (输出):", "浏览...", self.choose_output)
        path_layout.addLayout(r1)
        path_layout.addLayout(r2)
        path_layout.addLayout(r3)

        scan_row = QHBoxLayout()
        scan_row.addStretch()
        self.btn_scan = QPushButton("扫描配对")
        self.btn_scan.setToolTip("根据文件名（不含扩展名）自动配对")
        self.btn_scan.clicked.connect(self.scan_pairs)
        scan_row.addWidget(self.btn_scan)
        path_layout.addLayout(scan_row)

        main_layout.addWidget(path_group)

        # ===== 中间分割区 =====
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：配对列表
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QLabel("配对列表（点击预览）："))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["序号", "文件名", "扩展名", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        list_layout.addWidget(self.table)
        splitter.addWidget(list_widget)

        # 右侧：预览区
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(QLabel("图像预览："))

        src_preview_group = QGroupBox("Source (原图)")
        src_preview_layout = QVBoxLayout(src_preview_group)
        self.lbl_src_preview = QLabel("未选择")
        self.lbl_src_preview.setAlignment(Qt.AlignCenter)
        self.lbl_src_preview.setMinimumSize(300, 200)
        self.lbl_src_preview.setStyleSheet("background-color: #2d2d2d; color: #aaaaaa;")
        src_preview_layout.addWidget(self.lbl_src_preview)
        preview_layout.addWidget(src_preview_group)

        tar_preview_group = QGroupBox("Target (参考图)")
        tar_preview_layout = QVBoxLayout(tar_preview_group)
        self.lbl_tar_preview = QLabel("未选择")
        self.lbl_tar_preview.setAlignment(Qt.AlignCenter)
        self.lbl_tar_preview.setMinimumSize(300, 200)
        self.lbl_tar_preview.setStyleSheet("background-color: #2d2d2d; color: #aaaaaa;")
        tar_preview_layout.addWidget(self.lbl_tar_preview)
        preview_layout.addWidget(tar_preview_group)

        splitter.addWidget(preview_widget)
        splitter.setSizes([400, 700])
        main_layout.addWidget(splitter, stretch=1)

        # ===== 底部控制区 =====
        ctrl_group = QGroupBox("处理选项")
        ctrl_layout = QHBoxLayout(ctrl_group)

        ctrl_layout.addWidget(QLabel("进程数:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 16)
        import multiprocessing
        self.thread_spin.setValue(min(multiprocessing.cpu_count(), 8))
        ctrl_layout.addWidget(self.thread_spin)

        ctrl_layout.addSpacing(20)

        # LUT 导出选项
        self.chk_export_lut = QCheckBox("导出 3D LUT (.cube)")
        self.chk_export_lut.setToolTip("为每对图像生成一个 .cube 文件，可在 PS/PR/达芬奇等软件中复用")
        ctrl_layout.addWidget(self.chk_export_lut)

        ctrl_layout.addWidget(QLabel("精度:"))
        self.lut_size_spin = QSpinBox()
        self.lut_size_spin.setRange(9, 129)
        self.lut_size_spin.setValue(33)
        self.lut_size_spin.setSingleStep(2)
        self.lut_size_spin.setToolTip("LUT 网格尺寸，越大越精确但文件越大。推荐 33 或 64。")
        ctrl_layout.addWidget(self.lut_size_spin)

        ctrl_layout.addStretch()

        self.btn_start = QPushButton("开始处理")
        self.btn_start.setStyleSheet("QPushButton { padding: 8px 28px; font-weight: bold; }")
        self.btn_start.clicked.connect(self.start_processing)
        ctrl_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("中断")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        ctrl_layout.addWidget(self.btn_stop)

        main_layout.addWidget(ctrl_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        main_layout.addWidget(QLabel("日志:"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.document().setMaximumBlockCount(500)
        main_layout.addWidget(self.log_edit)

        self.statusBar().showMessage("请选择文件夹并点击「扫描配对」")

    def choose_source(self):
        d = QFileDialog.getExistingDirectory(self, "选择 Source 文件夹")
        if d:
            self.src_edit.setText(d)

    def choose_target(self):
        d = QFileDialog.getExistingDirectory(self, "选择 Target 文件夹")
        if d:
            self.tar_edit.setText(d)

    def choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择 Output 文件夹")
        if d:
            self.out_edit.setText(d)

    def scan_pairs(self):
        src = self.src_edit.text().strip()
        tar = self.tar_edit.text().strip()

        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "路径错误", "请选择有效的 Source 文件夹。")
            return
        if not tar or not os.path.isdir(tar):
            QMessageBox.warning(self, "路径错误", "请选择有效的 Target 文件夹。")
            return

        src_path = Path(src)
        tar_path = Path(tar)
        img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

        source_files = {f.stem: f for f in src_path.iterdir() if f.is_file() and f.suffix.lower() in img_exts}
        target_files = {f.stem: f for f in tar_path.iterdir() if f.is_file() and f.suffix.lower() in img_exts}

        self.pairs = []
        for stem in sorted(source_files.keys()):
            if stem in target_files:
                self.pairs.append(PairItem(stem, source_files[stem], target_files[stem]))

        self.refresh_table()
        self.statusBar().showMessage(
            f"扫描完成：Source共 {len(source_files)} 张，Target共 {len(target_files)} 张，成功配对 {len(self.pairs)} 对"
        )
        self.log(f"扫描完成：成功配对 {len(self.pairs)} 对")

    def refresh_table(self):
        self.table.setRowCount(len(self.pairs))
        for i, pair in enumerate(self.pairs):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(pair.stem))
            self.table.setItem(i, 2, QTableWidgetItem(f"{pair.source_path.suffix} / {pair.target_path.suffix}"))
            self.table.setItem(i, 3, QTableWidgetItem(pair.status))

    def on_table_selection_changed(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if row < 0 or row >= len(self.pairs):
            return

        pair = self.pairs[row]
        self.statusBar().showMessage(
            f"预览: {pair.stem}  |  Source: {pair.source_path.name}  |  Target: {pair.target_path.name}"
        )

        src_img = cv2.imread(str(pair.source_path), cv2.IMREAD_COLOR)
        tar_img = cv2.imread(str(pair.target_path), cv2.IMREAD_COLOR)

        self._current_source_img = src_img
        self._current_target_img = tar_img

        self._set_preview(self.lbl_src_preview, src_img, "Source 加载失败")
        self._set_preview(self.lbl_tar_preview, tar_img, "Target 加载失败")

    def _set_preview(self, label: QLabel, img: Optional[np.ndarray], err_text: str):
        if img is None:
            label.setText(err_text)
            label.setPixmap(QPixmap())
            return
        pm = numpy_to_pixmap(img, max_size=500)
        if pm:
            label.setPixmap(pm)
            label.setText("")
        else:
            label.setText(err_text)
            label.setPixmap(QPixmap())

    def log(self, msg: str):
        self.log_edit.append(msg)

    def start_processing(self):
        if not self.pairs:
            QMessageBox.warning(self, "无配对", "请先点击「扫描配对」生成处理列表。")
            return

        out = self.out_edit.text().strip()
        if not out:
            QMessageBox.warning(self, "路径错误", "请选择 Output 文件夹。")
            return

        for pair in self.pairs:
            pair.status = "待处理"
        self.refresh_table()
        self.progress.setValue(0)
        self.log_edit.clear()
        self.btn_start.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)

        export_lut = self.chk_export_lut.isChecked()
        lut_size = self.lut_size_spin.value()

        self.worker = ProcessWorker(
            self.pairs, Path(out),
            self.thread_spin.value(),
            export_lut=export_lut,
            lut_size=lut_size
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.pair_status.connect(self.update_pair_status)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def update_pair_status(self, row: int, status: str):
        if 0 <= row < self.table.rowCount():
            self.table.item(row, 3).setText(status)

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
        self.btn_start.setEnabled(True)
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def on_finished(self, success: bool, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.critical(self, "错误/中断", msg)


def main():
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
