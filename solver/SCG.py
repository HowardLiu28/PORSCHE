import cv2
import numpy as np
import glob
import os


def FreCom(img):
    h, w = img.shape[:2]
    img_dct = np.zeros((h, w, 3))
    for i in range(3):
        img_ = img[:, :, i]  # 获取rgb通道中的一个
        img_ = np.float32(img_)  # 将数值精度调整为32位浮点型
        img_dct[:, :, i] = cv2.dct(img_)  # 使用dct获得img的频域图像

    return img_dct


def Matching(img, reference, alpha=0.2, beta=1):
    theta = np.random.uniform(alpha, beta)
    h, w = img.shape[:2]
    img_dct = FreCom(img)

    mask = np.zeros((h, w, 3))
    # v1 = int(min(h, w) * 0.005)  # 低中频划分
    # v2 = int(min(h, w) * 0.7)  # 中高频划分
    v1 = min(h, w) * 0.05  # 低中频划分
    v2 = min(h, w) * 0.7  # 中高频划分
    v3 = min(h, w)
    # 简便带通滤波器设计
    for x in range(h):
        for y in range(w):
            if (max(x, y) <= v1):
                mask[x][y] = 1 - max(x, y) / v1 * 0.95
            elif (v1 < max(x, y) <= v2):
                mask[x][y] = 0.01
            elif (v2 <= max(x, y) <= v3):
                mask[x][y] = 0.01   # (max(x, y) - v2) / (v3 - v2) * 0.3
            else:
                mask[x][y] = 0.5
    n_mask = 1 - mask
    # 划分为因果部分和非因果部分
    non_img_dct = img_dct * mask
    cal_img_dct = img_dct * n_mask

    # 非因果部分随即变换
    # m, n = non_img_dct.shape[0], non_img_dct.shape[1]
    # ref_dct = np.zeros((m, n, 3))
    # ref_dct[:, :, 0] = non_img_dct[:, :, 0] * (1 + np.random.randn())
    # ref_dct[:, :, 1] = non_img_dct[:, :, 1] * (1 + np.random.randn())
    # ref_dct[:, :, 2] = non_img_dct[:, :, 2] * (1 + np.random.randn())

    # 重新组合
    # img_fc = ref_dct + cal_img_dct
    img_fc = cal_img_dct

    img_out = np.zeros((h, w, 3))
    for i in range(3):
        img_ = img_fc[:, :, i]  # 获取rgb通道中的一个
        img_out[:, :, i] = cv2.idct(img_).clip(0, 255)  # 使用dct获得img的频域图像

    return img_out


if __name__ == '__main__':

    print('start')
    img_path = r"dataset\rgbir\bounding_box_train"
    save_path = r"dataset\rgbir\bounding_box_train4"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_lists = glob.glob(img_path + '/*.jpg')
    img_basenames = []

    # 遍历所有的图片，取图片名
    for item in img_lists:
        img_basenames.append(os.path.basename(item))
    all_num = len(img_lists)

    i = 0
    for img_n, img_p in zip(img_basenames, img_lists):
        img = cv2.imread(img_p)
        roi_img = img[0:128, 0:256]
        h1, w1 = roi_img.shape[:2]
        # 如果 长宽 不是偶数 -> 缩放成偶数
        if h1 % 2 != 0 or w1 % 2 != 0:
            roi_img = cv2.resize(roi_img, (w1 - w1 % 2, h1 - h1 % 2), interpolation=cv2.INTER_AREA)

        refrence = np.ones_like(roi_img)
        refrence[:, :, 0] = refrence[:, :, 0] * np.random.randint(0, 255)
        refrence[:, :, 1] = refrence[:, :, 1] * np.random.randint(0, 255)
        refrence[:, :, 2] = refrence[:, :, 2] * np.random.randint(0, 255)
        np.random.randn()

        img_matched = Matching(roi_img, refrence)
        img[0:128, 0:256] = img_matched
        cv2.imwrite(save_path + '/' + img_n, img)
        i += 1
        print('[' + str(i) + '/' + str(all_num) + ']' + '\t' + img_n)
