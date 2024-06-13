from picamera2 import Picamera2, Preview
import cv2
import face_recognition
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 初始化当前识别的名字为“未知”
currentname = "Unknown"
# 加载 encodings.pickle 文件
encodingsP = "encodings.pickle"

# 加载已知的面部和嵌入
print("[INFO] 正在加载 encodings + 人脸检测器...")
try:
    with open(encodingsP, "rb") as f:
        data = pickle.load(f)
    print("[INFO] encodings 加载成功")
except Exception as e:
    print(f"[ERROR] 无法加载 encodings: {e}")
    exit()

# 初始化摄像头
print("[INFO] 初始化摄像头...")
try:
    picam2 = Picamera2()    
    # 创建配置
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)

    picam2.start()
    
    time.sleep(2.0)
    print("[INFO] 摄像头初始化成功")
except Exception as e:
    print(f"[ERROR] 摄像头初始化失败: {e}")
    exit()

# 创建窗口
cv2.namedWindow("Face Recognition Running", cv2.WINDOW_NORMAL)
print("[INFO] 创建窗口之后")

# 开始 FPS 计数器
fps_start_time = time.time()
print("[INFO] 开始 FPS 计数器之后")
frame_count = 0

# 加载中文字体
font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # 确保字体文件存在
try:
    font = ImageFont.truetype(font_path, 30)  # 字体大小调整为 30
except Exception as e:
    print(f"[ERROR] 无法加载字体: {e}")
    exit()

while True:
    try:
        frame_start_time = time.time()
        
        # 捕获一帧图像
        frame = picam2.capture_array()
        capture_time = time.time()
        print(f"[TIME] 捕获图像时间: {capture_time - frame_start_time:.4f} 秒")
        
        # 将图像从 RGB 转换为 BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        conversion_time = time.time()
        print(f"[TIME] 图像转换时间: {conversion_time - capture_time:.4f} 秒")
        
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_conversion_time = time.time()
        print(f"[TIME] 灰度转换时间: {gray_conversion_time - conversion_time:.4f} 秒")

        # 检测人脸框
        boxes = face_recognition.face_locations(frame_gray)
        detection_time = time.time()
        print(f"[TIME] 人脸检测时间: {detection_time - gray_conversion_time:.4f} 秒")
        print(f"[INFO] 检测到 {len(boxes)} 张人脸")

        # 初始化时间变量
        encoding_time = detection_time
        draw_time = detection_time

        # 为每个人脸边界框计算人脸嵌入
        if len(boxes) > 0:
            try:
                encodings = face_recognition.face_encodings(frame_bgr, boxes)
                encoding_time = time.time()
                print(f"[TIME] 计算人脸嵌入时间: {encoding_time - detection_time:.4f} 秒")
                print(f"[INFO] 计算了 {len(encodings)} 个人脸嵌入")
                
                if len(encodings) == 0:
                    print("[WARNING] 未计算到人脸嵌入，跳过当前帧")
                    continue
            except Exception as e:
                print(f"[ERROR] 计算人脸嵌入时发生错误: {e}")
                continue  # 跳过当前帧继续下一帧

            names = []

            # 遍历人脸嵌入
            for encoding in encodings:
                # 尝试将输入图像中的每个人脸与我们已知的嵌入进行匹配
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                print(f"[INFO] 找到匹配: {matches}")
                name = "未知"  # 如果未识别人脸，则打印“未知”

                # 检查是否找到匹配项
                if True in matches:
                    # 找到所有匹配人脸的索引，然后初始化一个字典来计算每个人脸被匹配的总次数
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # 遍历匹配的索引并维护每个被识别人脸的计数
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # 确定得到票数最多的已识别人脸（注意：在不太可能的平局情况下，Python 将选择字典中的第一个条目）
                    name = max(counts, key=counts.get)

                    # 如果识别到数据集中的人，则在屏幕上打印他们的姓名
                    if currentname != name:
                        currentname = name
                        print(f"[INFO] 识别到: {currentname}")

                # 更新姓名列表
                names.append(name)

            draw_start_time = time.time()
            # 将 OpenCV 图像转换为 PIL 图像
            pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # 遍历已识别人脸
            for ((top, right, bottom, left), name) in zip(boxes, names):
                try:
                    print(f"[INFO] 绘制人脸框和姓名: {name}, 框: {top, right, bottom, left}")
                    # 在图像上绘制预测的人脸姓名 - 颜色为绿色
                    draw.rectangle([(left, top), (right, bottom)], outline=(255, 0, 0), width=2)  # 红色边框
                    y = top - 30 if top - 30 > 30 else top + 30
                    draw.text((left, y), name, font=font, fill=(0, 255, 0))  # 绿色字体
                except Exception as e:
                    print(f"[ERROR] 绘制人脸框和姓名时出错: {e}")
            draw_time = time.time()
            print(f"[TIME] 绘制时间: {draw_time - draw_start_time:.4f} 秒")
            
            # 将 PIL 图像转换回 OpenCV 图像
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 在屏幕上显示图像
        cv2.imshow("Face Recognition Running", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        waitkey_time = time.time()
        print(f"[TIME] waitKey 时间: {waitkey_time - draw_time:.4f} 秒")
        print("[INFO] 调用 waitKey 完成")

        # 按 'q' 键退出
        if key == ord("q"):
            break

        # 更新 FPS 计数器
        frame_count += 1
        if frame_count >= 10:
            elapsed_time = time.time() - fps_start_time
            fps = frame_count / elapsed_time
            print(f"[INFO] 每秒帧数 (FPS): {fps:.2f}")
            frame_count = 0
            fps_start_time = time.time()

        frame_end_time = time.time()
        print(f"[TIME] 处理一帧总时间: {frame_end_time - frame_start_time:.4f} 秒")

    except Exception as e:
        print(f"[ERROR] 在主循环中发生错误: {e}")
        break

# 清理工作
cv2.destroyAllWindows()
picam2.stop()
print("[INFO] 清理完成，程序退出")
