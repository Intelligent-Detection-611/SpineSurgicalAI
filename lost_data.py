import pydicom
import numpy as np
from PIL import Image
import math

def calculate_ssim(img1, img2, data_range=255, C1=0.01**2, C2=0.03**2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    SSIM = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    return SSIM

def calculate_entropy(image_array):
    image_array = image_array.astype(np.uint8)
    hist, _ = np.histogram(image_array.flatten(), bins=256, range=[0, 255])
    prob_distribution = hist / np.sum(hist)
    prob_distribution = prob_distribution[prob_distribution > 0]
    entropy = -np.sum(prob_distribution * np.array([math.log2(p) for p in prob_distribution]))
    return entropy

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
            img = (img - img_min) / (img_max - img_min) * 255.0

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
            entropy_dicom = calculate_entropy(pixels_dicom_scaled)
            print(f"原始 DICOM 像素数组 (窗宽窗位调整后) 的香农熵: {entropy_dicom:.4f}")
            entropy_png = calculate_entropy(pixels_png)
            print(f"PNG 图像像素数组的香农熵: {entropy_png:.4f}")
            entropy_diff = entropy_dicom - entropy_png
            print(f"熵值差异 (DICOM - PNG): {entropy_diff:.4f}")
            ssim_index = calculate_ssim(pixels_dicom_scaled, pixels_png, data_range=255)
            print(f"原始 DICOM 像素数组 (窗宽窗位调整后) 与 PNG 图像像素数组的 SSIM: {ssim_index:.4f}")
        else:
            print("错误: 原始缩放后的 DICOM 像素数组和 PNG 像素数组形状不匹配，无法进行比较。")
