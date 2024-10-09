# 图像配准

import numpy as np
import cv2
from PIL import ImageGrab


def image_registration(
    query_img_path: str, query_img_scale: float = 1.0, threshold: int = 460
):

    # 读取拼接图片
    imageA = cv2.imread(query_img_path)
    imageA = cv2.resize(imageA, None, fx=query_img_scale, fy=query_img_scale)

    imageB = ImageGrab.grab()
    imageB = cv2.cvtColor(np.array(imageB), cv2.COLOR_RGB2BGR)

    # 读取图像
    img1 = imageA
    img2 = imageB

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 使用SIFT找到关键点和描述符
    kp1, descriptors1 = sift.detectAndCompute(img1, None)
    kp2, descriptors2 = sift.detectAndCompute(img2, None)

    # 创建匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比率测试
    good = []
    n_good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            n_good.append(n)
    if good:
        pts2 = np.float32([kp2[n.trainIdx].pt for m in n_good]).reshape(-1, 1, 2)
        mean_pt2 = np.mean(pts2, axis=0).ravel()

        # 剔除明显超出平均范围的点
        filtered_good = []
        for m, n in zip(good, n_good):
            pt = np.float32([kp2[n.trainIdx].pt])
            distance = np.linalg.norm(pt - mean_pt2)
            if distance < threshold:  # 这里的100是阈值，可以根据实际情况调整
                filtered_good.append([m])
        # 绘制图像
        img_matches = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, filtered_good, None, flags=2
        )
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(filtered_good) > 4:
            pts1 = np.float32([kp1[m[0].queryIdx].pt for m in filtered_good]).reshape(
                -1, 1, 2
            )
            pts2 = np.float32([kp2[m[0].trainIdx].pt for m in filtered_good]).reshape(
                -1, 1, 2
            )

            min_x1 = np.min(pts1[:, :, 0])
            max_x1 = np.max(pts1[:, :, 0])
            min_y1 = np.min(pts1[:, :, 1])
            max_y1 = np.max(pts1[:, :, 1])
            min_x2 = np.min(pts2[:, :, 0])
            max_x2 = np.max(pts2[:, :, 0])
            min_y2 = np.min(pts2[:, :, 1])
            max_y2 = np.max(pts2[:, :, 1])

            scale_x = (max_x2 - min_x2) / (max_x1 - min_x1)
            scale_y = (max_y2 - min_y2) / (max_y1 - min_y1)
            pts1_norm=pts1-[min_x1,min_y1]
            pts2_norm=pts2-[min_x2,min_y2]
            H, masked = cv2.findHomography(pts1_norm, pts2_norm, cv2.RANSAC, 5.0)

            # 使用单应性矩阵变换查询图像
            height, width, _ = img1.shape
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)

            img1_transformed = cv2.warpPerspective(img1, H, (new_width,new_height))
            #img1_transformed = cv2.resize(img1, (new_width, new_height))
            #print(img1_transformed.shape)
            cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
            cv2.imshow("Matches", img1_transformed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 将变换后的查询图像定位到全景图像中
            img2[
                int(min_y2 - min_y1) : int(min_y2 - min_y1) + new_height,
                int(min_x2 - min_x1) : int(min_x2 - min_x1) + new_width,
            ] = img1_transformed[:, :]

            # 显示图像
            cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
            cv2.imshow("Matches", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    image_registration(query_img_path=r"D:\work\ComfyUI\output\ComfyUI_00070.png", query_img_scale=0.05)
