import cv2

# 打开摄像头（0表示默认摄像头，其他数字可以选择其他摄像头）
cap = cv2.VideoCapture(0)

# 检查摄像头是否打开成功
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取摄像头的一帧
    ret, frame = cap.read()

    # 如果成功读取，显示这帧影像
    if ret:
        cv2.imshow('Camera Feed', frame)
    else:
        print("无法读取帧")
        break

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
 