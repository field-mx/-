import cv2
from main import resize_image

# 创建 SIFT 对象
sift = cv2.SIFT_create()
image = resize_image('room1.jpg', 0.3)
# 读取图像
#image = cv2.imread('room1.jpg', cv2.IMREAD_GRAYSCALE)

# 检测关键点并计算描述符
keypoints, descriptors = sift.detectAndCompute(image, None)

# 在图像上绘制关键点
output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示结果
cv2.namedWindow('SIFT Keypoints', cv2.WINDOW_NORMAL)
cv2.imshow('SIFT Keypoints', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
