import cv2
import numpy as np


def resize_image(input_image_path, scale=None, target_width=None, target_height=None):
    """
    等比缩放图像并保存。

    参数:
    input_image_path (str): 输入图像的路径。
    output_image_path (str): 输出缩放图像的路径。
    scale (float): 缩放比例，如 0.5 表示缩小一半。如果提供了比例，将忽略目标宽度和高度。
    target_width (int): 目标宽度，优先级低于比例缩放。
    target_height (int): 目标高度，优先级低于比例缩放。
    """
    # 读取图像
    img = cv2.imread(input_image_path)

    # 检查图像是否成功读取
    if img is None:
        raise ValueError("无法读取输入图像，请检查文件路径。")

    original_height, original_width = img.shape[:2]

    if scale is not None:
        # 等比缩放图像
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    elif target_width is not None and target_height is not None:
        # 根据目标尺寸等比缩放图像，选择最小的缩放因子
        scale_width = target_width / original_width
        scale_height = target_height / original_height
        scale = min(scale_width, scale_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    else:
        raise ValueError("请提供缩放比例或目标尺寸（宽度和高度）。")

    # 调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img


class FeatureMatcher:
    def __init__(self, detector_type='ORB', matcher_type='BF', use_cross_check=True):
        """
        初始化特征匹配器和检测器。

        参数:
        - detector_type: 特征检测器类型，'ORB', 'SIFT', 'SURF' 等
        - matcher_type: 匹配器类型，'BF' 使用暴力匹配器，'FLANN' 使用FLANN匹配器
        - use_cross_check: 是否启用交叉检查，仅对BFMatcher有效
        """
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.use_cross_check = use_cross_check
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()

    def _create_detector(self):
        """根据选择的检测器类型创建特征检测器。"""
        if self.detector_type == 'ORB':
            return cv2.ORB_create()
        elif self.detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif self.detector_type == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError("Unsupported detector type. Use 'ORB', 'SIFT', or 'SURF'.")

    def _create_matcher(self):
        """根据选择的匹配方法创建匹配器。"""
        if self.matcher_type == 'BF':
            norm_type = cv2.NORM_HAMMING if self.detector_type == 'ORB' else cv2.NORM_L2

            return cv2.BFMatcher(norm_type, crossCheck=False)

        elif self.matcher_type == 'FLANN':
            index_params = dict(algorithm=6, table_number=6, key_size=12,
                                multi_probe_level=1) if self.detector_type == 'ORB' else dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Unsupported matcher type. Use 'BF' or 'FLANN'.")

    def detect_and_compute(self, img):
        """
        检测图像中的特征点并计算描述符。

        参数:
        - img: 输入图像

        返回:
        - keypoints: 检测到的关键点
        - descriptors: 计算的描述符
        """
        keypoints, descriptors = self.detector.detectAndCompute(img, None)
        return keypoints, descriptors

    def match(self, descriptors1, descriptors2):
        """
        匹配两个图像的描述符。

        参数:
        - descriptors1: 第一张图像的描述符
        - descriptors2: 第二张图像的描述符

        返回:
        - good_matches: 过滤后的匹配结果
        """
        global good_matches
        if self.matcher_type == 'BF':
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_matches.append(m)



        elif self.matcher_type == 'FLANN':
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        return good_matches

    def get_homo(self, good_matches):
        # 单应性矩阵函数
        min_matchs = 8
        if len(good_matches) > min_matchs:
            img1_pts = []
            img2_pts = []

            for m in good_matches:
                img1_pts.append(keypoints1[m.queryIdx].pt)
                img2_pts.append(keypoints2[m.trainIdx].pt)

            img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
            img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

            # 获取但应性矩阵和掩码
            H_martix, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
            return H_martix
        else:
            print('error: 不够8个特征点')

    def stitch_image(self, img1, img2, H):
        """

        图像拼接：
        1.获取四个角点
        2.对图像变换，旋转平移
        3.创建画布拼接

        :param img1:
        :param img2:
        :param H:单应性矩阵
        :return:拼接图片

        """

        # 原始图形高度宽度,四个角点
        h_1, w_1 = img1.shape[:2]
        h_2, w_2 = img2.shape[:2]

        img1_dims = np.float32([[0, 0], [0, h_1], [w_1, h_1], [w_1, 0]]).reshape(-1, 1, 2)
        img2_dims = np.float32([[0, 0], [0, h_2], [w_2, h_2], [w_2, 0]]).reshape(-1, 1, 2)

        # 变化后的角点 = 之前的 * dyx矩阵
        img1_transform = cv2.perspectiveTransform(img1_dims, H)
        # print(img1_transform)
        # 拼接定图和动图的角点
        result_dims = np.concatenate((img2_dims, img1_transform), axis=0)
        print(result_dims)

        [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

        # 平移矩阵
        trans_dist = [-x_min, -y_min]

        # 构造平移矩阵并将其转换为 NumPy 数组
        transform_array = np.array(
            [[1, 0, trans_dist[0]],
             [0, 1, trans_dist[1]],
             [0, 0, 1]]
        )

        # 确保 H 也是一个 NumPy 数组
        H = np.array(H)

        # 进行矩阵乘法（矩阵变换）
        final_transform = np.dot(transform_array, H)

        # 应用透视变换,拼接图像
        result_img = cv2.warpPerspective(img1, final_transform, (x_max - x_min, y_max - y_min))
        result_img[trans_dist[1]:trans_dist[1] + h_2,
        trans_dist[0]:trans_dist[0] + w_2] = img2

        return result_img

    def draw_matches(self, img1, keypoints1, img2, keypoints2, matches):
        """
        绘制并显示匹配结果。

        参数:
        - img1: 第一张图像
        - keypoints1: 第一张图像的关键点
        - img2: 第二张图像
        - keypoints2: 第二张图像的关键点
        - matches: 匹配结果
        """
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def resize_to_smallest(image1_path, image2_path, output1_path, output2_path):
    # 读取两张图片
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 获取两张图片的尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 找到较小的尺寸
    new_width = min(w1, w2)
    new_height = min(h1, h2)

    # 调整两张图片的大小
    img1_resized = cv2.resize(img1, (new_width, new_height))
    img2_resized = cv2.resize(img2, (new_width, new_height))

    # 保存调整后的图片
    cv2.imwrite(output1_path, img1_resized)
    cv2.imwrite(output2_path, img2_resized)

    print(f"图片已保存为 '{output1_path}' 和 '{output2_path}'")


# 导入图片，模糊因子选择：0.1
img1 = resize_image('selected_roi.jpg', 1)
img2 = resize_image('selected_roi34.jpg', 1)

#resize_to_smallest('image1.jpg', 'image2.jpg', 'output_image1.jpg', 'output_image2.jpg')

# 创建特征
# 特征点标注算法选择：SIFT/ORB/SURF
# 特征匹配算法：FLANN/BF
matcher = FeatureMatcher(detector_type='SIFT', matcher_type='BF')
keypoints1, descriptors1 = matcher.detect_and_compute(img1)
keypoints2, descriptors2 = matcher.detect_and_compute(img2)

# 特征匹配
good_matches = matcher.match(descriptors1, descriptors2)

# 使用 cv2.drawMatches 绘制匹配结果
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# 获取矩阵
H = matcher.get_homo(good_matches)
# 图像拼接
result_img = matcher.stitch_image(img1, img2, H)

# 显示匹配结果
cv2.namedWindow('Feature Matches', cv2.WINDOW_NORMAL)
cv2.imshow('Feature Matches', result_img)
cv2.imwrite( '1234.jpg', result_img )
cv2.waitKey(0)
cv2.destroyAllWindows()
