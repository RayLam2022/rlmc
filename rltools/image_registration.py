# 图像配准

import numpy as np
import cv2
from PIL import ImageGrab
from sklearn.cluster import DBSCAN


def image_registration(
    query_img_path: str, query_img_scale: float = 1.0, threshold: float = 1.0, min_samples: int =4,is_show: bool = True
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
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)##

    # 应用比率测试
    good = []
    n_good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            n_good.append(n)
    if not good:
        print('没有匹配点')
        return None
    else:
        # 绘制图像
        pts2 = np.float32([kp2[n.trainIdx].pt for n in n_good]).reshape(-1, 2)
        dbscan = DBSCAN(eps=threshold, min_samples=min_samples)
 
        # 对数据集进行聚类
        labels = dbscan.fit_predict(pts2)
        print('labels:',labels)
        #is_core_sample = dbscan.core_sample_indices_
        # 剔除离群点，保留标签为-1的点
        # pts2 = pts2[~outliers.any(axis=1)]
        filtered_good = []
        for idx, (m, n) in enumerate(zip(good, n_good)):
            if labels[idx]>=0:
                filtered_good.append([m])


        img_matches = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, filtered_good, None, flags=2
        )
        if is_show:
            cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
            cv2.imshow("Matches", img_matches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if len(filtered_good) <= 4:
            print('匹配点小于5')
            return None
        else:
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
            if is_show:
                cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
                cv2.imshow("Matches", img1_transformed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # 将变换后的查询图像定位到全景图像中
            try:
                img2[
                    int(min_y2-min_y1*scale_y) : int(min_y2-min_y1* scale_y) + new_height,
                    int(min_x2-min_x1*scale_x) : int(min_x2-min_x1*scale_x) + new_width,
                ] = img1_transformed
            except:
                print("out of range，可能有异常点导致图片切片过大")
                return None

            # 显示图像
            else:
                if is_show:
                    cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
                    cv2.imshow("Matches", img2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_registration(query_img_path=r"D:\work\ComfyUI\output\ComfyUI_00070.png", query_img_scale=0.2, threshold=22.0, min_samples=3, is_show=True)
