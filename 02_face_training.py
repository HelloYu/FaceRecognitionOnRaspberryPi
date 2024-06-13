# 导入必要的包
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# 图片位于 dataset 文件夹中
print("[INFO] 开始处理人脸...")
imagePaths = list(paths.list_images("dataset"))

# 初始化已知的编码和名字列表
knownEncodings = []
knownNames = []

# 遍历图片路径
for (i, imagePath) in enumerate(imagePaths):
    # 从图片路径中提取人物名字
    print("[INFO] 处理图像 {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    print(f"[DEBUG] 当前处理的图片路径: {imagePath}")
    print(f"[DEBUG] 当前处理的名字: {name}")

    # 读取输入图像并将其从 BGR (OpenCV 读取顺序) 转换为 RGB (dlib 使用的顺序)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"[DEBUG] 图像读取并转换为 RGB 格式")

    # 检测输入图像中每张人脸的 (x, y)-坐标的边界框
    boxes = face_recognition.face_locations(rgb, model="hog")
    print(f"[DEBUG] 检测到 {len(boxes)} 张人脸")

    # 计算人脸的嵌入（即人脸编码）
    encodings = face_recognition.face_encodings(rgb, boxes)
    print(f"[DEBUG] 计算到 {len(encodings)} 个编码")

    # 遍历每一个编码
    for encoding in encodings:
        # 将每个编码和名字添加到已知的编码和名字列表中
        knownEncodings.append(encoding)
        knownNames.append(name)
        print(f"[DEBUG] 已知编码和名字列表更新")

# 将人脸编码和名字序列化到磁盘中
print("[INFO] 序列化编码...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] 编码序列化完成")
