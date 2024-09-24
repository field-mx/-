import cv2


def select_roi(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 显示图像并手动选择ROI
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)

    # 获取ROI的坐标和尺寸
    x, y, w, h = roi

    # 截取ROI
    roi_image = image[y:y + h, x:x + w]

    # 调整窗口大小以适应图像
    cv2.namedWindow("Selected ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Selected ROI", roi_image)
    cv2.resizeWindow("Selected ROI", w, h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存截取的ROI
    cv2.imwrite('1234_resized.jpg', roi_image)
    print("截取的ROI已保存为 'selected_roi.jpg'")


# 调用函数并传入图像路径
select_roi('1234.jpg')
