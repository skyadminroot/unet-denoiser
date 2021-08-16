import os
import time

import cv2
import fitz
from tqdm import tqdm


def pdf_image(pdf_path, dest_path, zoom_x, zoom_y, rotation_angle):
    # 打开PDF文件
    pdf = fitz.open(pdf_path)
    # 逐页读取PDF
    total_count = pdf.pageCount
    with tqdm(total=total_count) as progress_bar:
        for pg in range(0, total_count):
            page = pdf[pg]
            # 设置缩放和旋转系数
            trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
            pm = page.getPixmap(matrix=trans, alpha=False)
            # 开始写图像
            pm.writePNG(dest_path + str(time.time_ns()) + ".png")
            progress_bar.update(1)
    pdf.close()


def convertAllPdf(dir):
    if not os.path.exists(dir):
        print(dir, "not exist")
        return
    for file_name in os.listdir(dir):
        path = os.path.join(dir, file_name)
        if os.path.isdir(path):
            continue
        dest_path = 'pdf2image/' + file_name.replace('.pdf', '/')
        os.makedirs(dest_path, exist_ok=True)
        pdf_image(path, dest_path, 3, 3, 0)
    pass


def resize_all_image(image_path, dest_path, width, height):
    if not os.path.exists(image_path):
        return
    bitmap = cv2.imread(image_path)
    target_bitmap = cv2.resize(bitmap, (width, height))
    cv2.imwrite(dest_path, target_bitmap)


def resize_all_image_on_dictionary(dir):
    if not os.path.exists(dir):
        return
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isdir(file_path):
            child_list = os.listdir(file_path)
            with tqdm(total=len(child_list)) as progress_bar:
                for child_name in child_list:
                    child_path = os.path.join(file_path, child_name)
                    dest_path_parent = "./pdf2image-dev/" + file_name + "/"
                    os.makedirs(dest_path_parent, exist_ok=True)
                    dest_path = dest_path_parent + child_name
                    resize_all_image(child_path, dest_path, 1360, 1870)
                    progress_bar.update(1)


if __name__ == '__main__':
    # convertAllPdf('pdf')
    resize_all_image_on_dictionary('./pdf2image')
    pass
