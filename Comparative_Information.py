import pydicom
import numpy as np
from PIL import Image
import math
from scipy.stats import pearsonr
import os

# 熵
def calculate_entropy(image_array):
    if image_array.dtype != np.uint8:
        # 自动根据数据范围选择分箱数
        bins = min(4096, int(image_array.max() - image_array.min() + 1))
        hist, _ = np.histogram(image_array.flatten(), bins=bins)
    else:
        hist, _ = np.histogram(image_array.flatten(), bins=256, range=[0, 255])
    prob_distribution = hist / np.sum(hist)
    prob_distribution = prob_distribution[prob_distribution > 0]
    entropy = -np.sum(prob_distribution * np.array([math.log2(p) for p in prob_distribution]))
    return entropy

# 互信息
def calculate_mutual_information(img1, img2):
    # 互信息建议先对齐到相同shape和位深
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    # 分别归一化为256灰度级
    img1_scaled = ((img1_flat - img1_flat.min()) / (img1_flat.max() - img1_flat.min()) * 255).astype(np.uint8)
    img2_scaled = ((img2_flat - img2_flat.min()) / (img2_flat.max() - img2_flat.min()) * 255).astype(np.uint8)
    hist_2d, _, _ = np.histogram2d(img1_scaled, img2_scaled, bins=256)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    return mi

# 相关系数
def calculate_correlation_coefficient(img1, img2):
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    # 若灰度范围差太大，建议归一化到0-1比较稳妥
    img1_norm = (img1_flat - img1_flat.min()) / (img1_flat.max() - img1_flat.min())
    img2_norm = (img2_flat - img2_flat.min()) / (img2_flat.max() - img2_flat.min())
    corr, _ = pearsonr(img1_norm, img2_norm)
    return corr

# 只读取原始像素，不做任何处理
def load_raw_dicom_pixels(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array  # 保持原始类型和范围
        return pixel_array
    except Exception as e:
        print(f"读取 DICOM 原始像素时出错: {e}")
        return None

# 读取PNG（标准uint8灰度）
def load_png_image(png_path):
    try:
        img = Image.open(png_path).convert('L')
        return np.array(img)
    except Exception as e:
        print(f"读取 PNG 时出错: {e}")
        return None

if __name__ == "__main__":
    dicom_file = r"D:\task\github\SpineSurgicalAI\test_img\test1.dcm"
    png_file   = r"D:\task\github\SpineSurgicalAI\test_img\test1.png"  # dicom2png.py 生成的 PNG

    if not os.path.exists(dicom_file):
        print(f"错误: DICOM 文件不存在: {dicom_file}")
        exit(1)
    if not os.path.exists(png_file):
        print(f"错误: PNG 文件不存在: {png_file}")
        exit(1)

    # 读取数据
    pixels_dicom = load_raw_dicom_pixels(dicom_file)
    pixels_png = load_png_image(png_file)

    if pixels_dicom is not None and pixels_png is not None:
        # 尺寸需强制对齐
        if pixels_dicom.shape != pixels_png.shape:
            print(f"警告: DICOM与PNG尺寸不一致。将裁剪到最小公共shape。")
            min_shape = (min(pixels_dicom.shape[0], pixels_png.shape[0]),
                         min(pixels_dicom.shape[1], pixels_png.shape[1]))
            pixels_dicom = pixels_dicom[:min_shape[0], :min_shape[1]]
            pixels_png = pixels_png[:min_shape[0], :min_shape[1]]

        print("====== 指标对比（原始 DICOM 像素 vs PNG 灰度）======")
        # 熵
        entropy_dicom = calculate_entropy(pixels_dicom)
        entropy_png = calculate_entropy(pixels_png)
        print(f"熵 Entropy (DICOM原始, PNG): {entropy_dicom:.4f}, {entropy_png:.4f}")
        print(f"熵值差异 (DICOM原始 - PNG): {entropy_dicom - entropy_png:.4f}")

        # 互信息
        mi = calculate_mutual_information(pixels_dicom, pixels_png)
        print(f"互信息 MI: {mi:.4f}")

        # 相关系数
        cc = calculate_correlation_coefficient(pixels_dicom, pixels_png)
        print(f"相关系数 CC: {cc:.6f}")
    else:
        print("读取像素数据失败，无法比较。")
