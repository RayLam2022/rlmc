import cv2


# 鼠标事件回调函数
def display_coordinates(event, x, y, flags, param):
    # 获取全局变量image和image_copy
    global image, image_copy

    if event == cv2.EVENT_MOUSEMOVE:
        # 恢复原始图像
        image_copy = image.copy()
        # 在图像上显示坐标
        text = f"({x}, {y})"
        cv2.putText(
            image_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )
        # 画一个标记点
        cv2.circle(image_copy, (x, y), 4, (0, 0, 255), -1)
        # 更新显示的图像
        cv2.imshow("Image Coordinates", image_copy)


# 主程序
if __name__ == "__main__":
    # 读取图像（替换为你的图片路径）
    image_path = r"D:\work\rastra_rs\input\upload\OIP.jpg"
    image = cv2.imread(image_path)

    # 检查图像是否成功加载
    if image is None:
        print("无法加载图像，请检查路径")
        exit()

    # 创建图像副本
    image_copy = image.copy()

    # 创建窗口并绑定回调函数
    cv2.namedWindow("Image Coordinates")
    cv2.setMouseCallback("Image Coordinates", display_coordinates)

    # 初始显示图像
    cv2.imshow("Image Coordinates", image)

    # 按ESC退出
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break

    cv2.destroyAllWindows()
