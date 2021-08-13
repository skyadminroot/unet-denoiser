import os
import time

import fitz


def pdf_image(pdfPath, imgPath, zoom_x, zoom_y, rotation_angle):
    # 打开PDF文件
    pdf = fitz.open(pdfPath)
    # 逐页读取PDF
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
        pm = page.getPixmap(matrix=trans, alpha=False)
        # 开始写图像
        pm.writePNG(imgPath + str(time.time()) + ".png")
    pdf.close()


def convertAllPdf(dir):
    if not os.path.exists(dir):
        print(dir, "not exist")
        return
    for file_name in os.listdir(dir):
        path = os.path.join(dir, file_name)
        if os.path.isdir(path):
            continue
        pdf_image(path, 'pdf2image/', 3, 3, 0)
    pass


if __name__ == '__main__':
    convertAllPdf('pdf')
    pass
