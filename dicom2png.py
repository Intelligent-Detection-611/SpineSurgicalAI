import pydicom
from PIL import Image
import numpy as np
import os

def dicom_to_png(dicom_path, output_path):
    """
    将单个 DICOM 文件转换为 PNG 图像，高细节对比度模式（与 lost.py 一致）
    """
    try:
        ds = pydicom.dcmread(dicom_path)

        # 检查是否为彩色DICOM
        if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation == 'RGB':
            # 处理RGB彩色图像
            pixel_array = ds.pixel_array
            if pixel_array.dtype != np.uint8:
                pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixel_array, 'RGB')
        else:
            # 处理灰阶DICOM
            pixel_array = ds.pixel_array.astype(np.float32)

            # Rescale Slope/Intercept（CT常见）
            slope = float(ds.get("RescaleSlope", 1.0))
            intercept = float(ds.get("RescaleIntercept", 0.0))
            pixel_array = pixel_array * slope + intercept

            # 窗宽窗位
            wc = ds.get('WindowCenter', None)
            ww = ds.get('WindowWidth', None)
            if wc is not None and ww is not None:
                # 兼容 MultiValue 情况
                if isinstance(wc, pydicom.multival.MultiValue):
                    wc = float(wc[0])
                else:
                    wc = float(wc)
                if isinstance(ww, pydicom.multival.MultiValue):
                    ww = float(ww[0])
                else:
                    ww = float(ww)
                img_min = wc - ww / 2
                img_max = wc + ww / 2
                pixel_array = np.clip(pixel_array, img_min, img_max)
                # **窗宽窗位clip后再次归一化（与lost一致）**
                real_min = pixel_array.min()
                real_max = pixel_array.max()
                if real_max > real_min:
                    pixel_array = (pixel_array - real_min) / (real_max - real_min) * 255.0
                else:
                    pixel_array = np.zeros_like(pixel_array)
            else:
                # 没有窗宽窗位，直接全幅归一化
                real_min = pixel_array.min()
                real_max = pixel_array.max()
                if real_max > real_min:
                    pixel_array = (pixel_array - real_min) / (real_max - real_min) * 255.0
                else:
                    pixel_array = np.zeros_like(pixel_array)

            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)

            # PhotometricInterpretation 判断
            photometric = ds.get('PhotometricInterpretation', 'MONOCHROME2')
            if photometric == 'MONOCHROME1':
                pixel_array = 255 - pixel_array

            image = Image.fromarray(pixel_array, 'L')

        image.save(output_path, "PNG")
        print(f"成功转换 {dicom_path} 到 {output_path}")

    except Exception as e:
        print(f"转换 {dicom_path} 时出错: {e}")

# 示例用法：
if __name__ == "__main__":
    # 在这里修改为你要转换的实际 DICOM 文件路径
    your_dicom_file_path = r"D:\task\github\SpineSurgicalAI\test_img\test1.dcm"
    # 在这里定义你想要的输出 PNG 文件路径
    your_output_png_path = r"D:\task\github\SpineSurgicalAI\test_img\test1.png"

    if os.path.exists(your_dicom_file_path):
        dicom_to_png(your_dicom_file_path, your_output_png_path)
    else:
        print(f"错误: 找不到指定的 DICOM 文件: {your_dicom_file_path}")
