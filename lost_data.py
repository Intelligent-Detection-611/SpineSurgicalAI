import pydicom
import numpy as np
from PIL import Image
import math
from scipy.stats import pearsonr

# 1. 熵
def calculate_entropy(image_array):
    image_array = image_array.astype(np.uint8)
    hist, _ = np.histogram(image_array.flatten(), bins=256, range=[0, 255])
    prob_distribution = hist / np.sum(hist)
    prob_distribution = prob_distribution[prob_distribution > 0]
    entropy = -np.sum(prob_distribution * np.array([math.log2(p) for p in prob_distribution]))
    return entropy

# 2. 平均梯度
def calculate_average_gradient(image_array):
    image_array = image_array.astype(np.float32)
    gx = np.diff(image_array, axis=1)  # shape: (H, W-1)
    gy = np.diff(image_array, axis=0)  # shape: (H-1, W)
    ag = (np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2  # 取平均
    return ag

# 3. 标准差
def calculate_standard_deviation(image_array):
    return np.std(image_array)

# 4. PSNR
def calculate_psnr(img1, img2, data_range=255):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

# 5. MSE
def calculate_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

# 6. 互信息（基于联合直方图）
def calculate_mutual_information(img1, img2):
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    return mi

# 7. 相关系数
def calculate_correlation_coefficient(img1, img2):
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    corr, _ = pearsonr(img1_flat, img2_flat)
    return corr

# 8. 空间频率
def calculate_spatial_frequency(image_array):
    image_array = image_array.astype(np.float32)
    rf = np.sqrt(np.mean(np.diff(image_array, axis=1) ** 2))
    cf = np.sqrt(np.mean(np.diff(image_array, axis=0) ** 2))
    sf = np.sqrt(rf ** 2 + cf ** 2)
    return sf

# 9. SSIM
def calculate_ssim(img1, img2, data_range=255, window_size=7, K1=0.01, K2=0.03):
    assert img1.shape == img2.shape, "输入图像尺寸必须一致"
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 为简单起见，用全局均值和方差计算（没有移动窗口/高斯加权）
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

# dicom转png部分
def dicom_to_png_for_comparison(dicom_path, temp_png_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)

        # 1. Rescale
        slope = float(ds.get("RescaleSlope", 1.0))
        intercept = float(ds.get("RescaleIntercept", 0.0))
        img = img * slope + intercept

        # 2. Window center/width
        wc = ds.get('WindowCenter', None)
        ww = ds.get('WindowWidth', None)

        if wc is not None and ww is not None:
            if isinstance(wc, pydicom.multival.MultiValue): wc = float(wc[0])
            else: wc = float(wc)
            if isinstance(ww, pydicom.multival.MultiValue): ww = float(ww[0])
            else: ww = float(ww)
            img_min = wc - ww / 2
            img_max = wc + ww / 2
            img = np.clip(img, img_min, img_max)
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img_min = np.min(img)
            img_max = np.max(img)
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = np.zeros_like(img)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # 3. Photometric Interpretation
        if ds.get('PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
            img = 255 - img

        Image.fromarray(img).save(temp_png_path)
        return img, img

    except Exception as e:
        print(f"处理 {dicom_path} 时出错: {e}")
        return None, None

if __name__ == "__main__":
    dicom_file = r"D:\task\github\SpineSurgicalAI\test_img\test1.dcm"
    temp_png_file = r"D:\task\github\SpineSurgicalAI\test_img\temp_image.png"
    pixels_dicom_scaled, pixels_png = dicom_to_png_for_comparison(dicom_file, temp_png_file)
    if pixels_dicom_scaled is not None and pixels_png is not None:
        if pixels_dicom_scaled.shape == pixels_png.shape:
            print("====== 指标对比 ======")
            # 熵
            entropy_dicom = calculate_entropy(pixels_dicom_scaled)
            entropy_png = calculate_entropy(pixels_png)
            print(f"熵 Entropy (DICOM, PNG): {entropy_dicom:.4f}, {entropy_png:.4f}")
            print(f"熵值差异 (DICOM - PNG): {entropy_dicom - entropy_png:.4f}")

            # 平均梯度
            ag_dicom = calculate_average_gradient(pixels_dicom_scaled)
            ag_png = calculate_average_gradient(pixels_png)
            print(f"平均梯度 AG (DICOM, PNG): {ag_dicom:.4f}, {ag_png:.4f}")

            # 标准差
            sd_dicom = calculate_standard_deviation(pixels_dicom_scaled)
            sd_png = calculate_standard_deviation(pixels_png)
            print(f"标准差 SD (DICOM, PNG): {sd_dicom:.4f}, {sd_png:.4f}")

            # PSNR
            psnr = calculate_psnr(pixels_dicom_scaled, pixels_png)
            print(f"峰值信噪比 PSNR: {psnr:.4f} dB")

            # MSE
            mse = calculate_mse(pixels_dicom_scaled, pixels_png)
            print(f"均方误差 MSE: {mse:.6f}")

            # 互信息
            mi = calculate_mutual_information(pixels_dicom_scaled, pixels_png)
            print(f"互信息 MI: {mi:.4f}")

            # 相关系数
            cc = calculate_correlation_coefficient(pixels_dicom_scaled, pixels_png)
            print(f"相关系数 CC: {cc:.6f}")

            # 空间频率
            sf_dicom = calculate_spatial_frequency(pixels_dicom_scaled)
            sf_png = calculate_spatial_frequency(pixels_png)
            print(f"空间频率 SF (DICOM, PNG): {sf_dicom:.4f}, {sf_png:.4f}")

            # SSIM
            ssim_index = calculate_ssim(pixels_dicom_scaled, pixels_png, data_range=255)
            print(f"结构相似性 SSIM: {ssim_index:.6f}")
        else:
            print("错误: 原始缩放后的 DICOM 像素数组和 PNG 像素数组形状不匹配，无法进行比较。")
