from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import os

# 提示用户输入名字
name = input('请输入你的名字: ')

# 如果目录不存在，则创建目录
os.makedirs(f"dataset/{name}", exist_ok=True)

# 初始化摄像头
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (512, 304)})
picam2.configure(config)
picam2.start()

img_counter = 0

while True:
    # 捕获一帧图像
    frame = picam2.capture_array()

    # 将图像从 RGB 转换为 BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 显示图像
    cv2.imshow("Press Space to take a photo", frame_bgr)
    
    # 处理按键事件
    k = cv2.waitKey(1)
    if k % 256 == 27:  # 按下 ESC 键
        print("按下 ESC 键，关闭...")
        break
    elif k % 256 == 32:  # 按下空格键
        img_name = f"dataset/{name}/image_{img_counter}.jpg"
        # 保存图像
        cv2.imwrite(img_name, frame_bgr)
        print(f"{img_name} 已保存!")
        img_counter += 1

cv2.destroyAllWindows()
picam2.stop()
