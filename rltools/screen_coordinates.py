import pyautogui
import time
import sys

try:
    print("实时显示鼠标坐标 (按Ctrl+C退出)")
    while True:
        # 获取当前鼠标位置
        x, y = pyautogui.position()

        # 格式化输出坐标信息
        position_str = f"X: {x:4}, Y: {y:4}"

        # 在终端动态更新显示位置（不换行）
        sys.stdout.write("\r" + position_str)
        sys.stdout.flush()

        # 延迟0.1秒
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n程序已退出")
