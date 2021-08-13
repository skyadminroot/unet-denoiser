import os
import random
import shutil
import time

import cv2
import numpy
import numpy as np
from tqdm import tqdm


def readAllImageFiles(dir):
    if not os.path.exists(dir):
        return []
    filenames = os.listdir(dir)
    target_files = []
    for filename in filenames:
        file = os.path.join(dir, filename)
        if os.path.isdir(file):
            tmp = readAllImageFiles(file)
            for t in tmp:
                target_files.append(t)
        else:
            if filename == '.DS_Store':
                continue
            target_files.append(file)
    return target_files


def imWrite(path, img):
    try:
        parent = os.path.dirname(path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)
        pass


def rotate_bound(image, angle, scale):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


def overlayImage(bgp, mask_image):
    # resize
    if mask_image.shape[1] > bgp.shape[1]:
        tW = int(bgp.shape[1] - 100)
        tH = int(mask_image.shape[0] * tW / mask_image.shape[1])
        mask_image = cv2.resize(mask_image, (tW, tH))

    # resize
    # bg:96*384
    # mask:111*383
    if mask_image.shape[0] > bgp.shape[0]:
        tH = int(bgp.shape[0] - 10)
        tW = int(mask_image.shape[1] * tH / mask_image.shape[0])
        mask_image = cv2.resize(mask_image, (tW, tH))

    width = mask_image.shape[1]
    height = mask_image.shape[0]

    start_x = 0
    if bgp.shape[1] - width - 1 > 0:
        start_x = numpy.random.randint(low=0, high=bgp.shape[1] - width - 1, size=1)[0]

    start_y = 0
    if bgp.shape[0] - height - 1 > 0:
        start_y = numpy.random.randint(low=0, high=bgp.shape[0] - height - 1, size=1)[0]

    for x in range(width):
        for y in range(height):
            if mask_image[y, x] != 255:
                try:
                    bgp[start_y + y, start_x + x] = mask_image[y, x]
                except Exception as error:
                    print(error)

    return start_x, start_y, width, height


def copy_handwrite_images(bg_image, all_img_list, files, dest_mask_dir, dest_image_dir, progress_bar):
    global mImageCount
    bg = cv2.imread(bg_image, cv2.IMREAD_GRAYSCALE)
    for idx, val in enumerate(files):
        mImageCount += 1
        dest_mask_name = os.path.join(dest_mask_dir, str(mImageCount) + ".png")
        dest_image_name = os.path.join(dest_image_dir, str(mImageCount) + ".png")

        # 合成多个遮盖图片
        sample_size = int(len(files) * 0.8)
        if sample_size > 10:
            sample_size = 10
        random_image = random.sample(all_img_list, sample_size)
        mask_image = np.ones(bg.shape, bg.dtype) * 255
        for imgUrl in random_image:
            img = cv2.imread(imgUrl, cv2.IMREAD_GRAYSCALE)
            img = rotate_bound(img, random.randint(-180, 180), random.uniform(0.3, 0.6))
            overlayImage(mask_image, img)

        bgp = bg.copy()
        start_x, start_y, width, height = overlayImage(bgp, mask_image)
        imWrite(dest_image_name, bgp)

        # resize mask_image
        # 此时mask_image尺寸必然小于bgp
        w = bgp.shape[1]
        h = bgp.shape[0]
        target_mask = np.ones((h, w), dtype=np.uint8) * 0
        tmp_mask_image = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        for x in range(width):
            for y in range(height):
                try:
                    target_mask[start_y + y, start_x + x] = tmp_mask_image[y, x]
                except Exception as error:
                    print(error)
                pass
        imWrite(dest_mask_name, target_mask)
        if progress_bar is not None:
            progress_bar.update(1)


mImageCount = 0


def generate_old():
    data_dir = os.path.join("./handwrite_data/", "data")
    # data_dir = os.path.join("./", "data")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    train_mask_dir = os.path.join(train_dir, "mask")
    if not os.path.exists(train_mask_dir):
        os.mkdir(train_mask_dir)
    compare_dir = os.path.join(data_dir, "compare")
    if not os.path.exists(compare_dir):
        os.mkdir(compare_dir)
    compare_mask_dir = os.path.join(compare_dir, "mask")
    if not os.path.exists(compare_mask_dir):
        os.mkdir(compare_mask_dir)
    compare_image_dir = os.path.join(compare_dir, "image")
    if not os.path.exists(compare_image_dir):
        os.mkdir(compare_image_dir)
    train_image_dir = os.path.join(train_dir, "image")
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    test_mask_dir = os.path.join(test_dir, "mask")
    if not os.path.exists(test_mask_dir):
        os.mkdir(test_mask_dir)
    test_image_dir = os.path.join(test_dir, "image")
    if not os.path.exists(test_image_dir):
        os.mkdir(test_image_dir)
    origin_files = readAllImageFiles("./handwrite")
    print(len(origin_files))
    # 百分之20做测试
    total = len(origin_files)
    train_num = int(total * 0.8)
    compare_num = int(total * 0.1)
    predict_files = origin_files[train_num: (total - compare_num)]
    train_files = origin_files[:train_num]
    compare_files = origin_files[(total - compare_num):]
    bgImage = "./handwrite_data_background/img_2.png"
    copy_handwrite_images(bgImage, predict_files, test_mask_dir, test_image_dir, None)
    copy_handwrite_images(bgImage, train_files, train_mask_dir, train_image_dir, None)
    copy_handwrite_images(bgImage, compare_files, compare_mask_dir, compare_image_dir, None)
    print("origin handwriting copy complete")


def generate_data(bg_dir, img_dir, random_mask_size):
    if not os.path.exists(bg_dir) or not os.path.exists(img_dir):
        print("resource dir not exist")
        return

    data_dir = "./data"

    train_image_dir = data_dir + "/train/image"
    train_mask_dir = data_dir + "/train/mask"

    test_image_dir = data_dir + "/test/image"
    test_mask_dir = data_dir + "/test/mask"

    # 所有背景图片集合
    bg_file_list = readAllImageFiles(bg_dir)
    # 所有手写图片集合
    img_file_list = readAllImageFiles(img_dir)

    # 测试用数据集
    test_bg_file_list = bg_file_list[0:5]
    # 训练用数据集
    train_bg_file_list = bg_file_list[5: 40]

    if len(bg_file_list) == 0 or len(img_file_list) == 0:
        print("miss resource file")
        return
    # 将M个背景图片映射到N个手写图片
    # 构建测试数据集
    with tqdm(total=len(test_bg_file_list) * random_mask_size) as progress_bar:
        for bgFile in test_bg_file_list:
            random.seed(time.time())
            img_slice = random.sample(img_file_list, random_mask_size)
            copy_handwrite_images(bgFile, img_file_list, img_slice, test_mask_dir, test_image_dir, progress_bar)

    # 构建训练数据集
    with tqdm(total=len(train_bg_file_list) * random_mask_size) as progress_bar:
        for bgFile in train_bg_file_list:
            random.seed(time.time())
            img_slice = random.sample(img_file_list, random_mask_size)
            copy_handwrite_images(bgFile, img_file_list, img_slice, train_mask_dir, train_image_dir, progress_bar)


snt = 0


def splitImageVertical(dir):
    global snt
    if not os.path.exists(dir):
        print(dir, "not exist")
        return
    for name in os.listdir(dir):
        if name == '.DS_Store':
            continue
        file = os.path.join(dir, name)
        # 过滤掉目录
        if os.path.isdir(file):
            continue
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("can't read file: " + file)
            continue
        image_height = image.shape[0]
        sub_image_height = int(image_height / 2)
        top_image = image[0:sub_image_height, :]
        bottom_image = image[sub_image_height:, :]

        snt += 1
        top_image_path = os.path.join("./casia", str(snt) + ".png")
        imWrite(top_image_path, top_image)

        snt += 1
        bottom_image_path = os.path.join("./casia", str(snt) + ".png")
        imWrite(bottom_image_path, bottom_image)


def resizeAllImage(dir):
    if not os.path.exists(dir):
        print(dir, "not exist")
        return
    files = readAllImageFiles(dir)
    for file in files:
        if 'DS_Store' in file:
            continue
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        w = int(image.shape[1] / 2)
        h = int(image.shape[0] / 2)
        image = cv2.resize(image, (w, h))
        imWrite(file, image)


if __name__ == '__main__':
    # generate_old()
    generate_data('./casia', '/Users/nutstore/awork/datasets/handwriting/casia/HWDB2.2Train_images', 5)
    resizeAllImage('./data')
    # splitImageVertical('./pdf2image')
    pass
