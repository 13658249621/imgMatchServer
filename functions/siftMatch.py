import os

import cv2

# 特征点匹配,已成功获取坐标
def feature_matching():
    tem = cv2.imread('/Users/admin/Downloads/模板.jpeg')
    tar = cv2.imread('/Users/admin/Downloads/目标.png')

    # 使用SIFT算法获取图像特征的关键点和描述符
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tem, None)
    kp2, des2 = sift.detectAndCompute(tar, None)

    height1, width2 = tem.shape[:2]
    print(height1, width2)

    # 定义FLANN匹配器
    indexParams = dict(algorithm=0, trees=10)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    # 使用KNN算法实现图像匹配，并对匹配结果排序
    matches = flann.knnMatch(des1, des2, k=2)



    # 去除错误匹配，0.5是系数，系数大小不同，匹配的结果页不同
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.25 * n.distance:
            goodMatches.append(m)

    # 获取某个点的坐标位置
    # index是获取匹配结果的中位数
    index = int(len(goodMatches) / 2)
    # queryIdx是目标图像的描述符索引
    x, y = kp1[goodMatches[index].queryIdx].pt
    print(x, y)
    # 将坐标位置勾画在2.png图片上，并显示
    cv2.rectangle(tem, (int(x), int(y)), (int(x) + 30, int(y) + 30), (0, 0, 255), 5)
    cv2.imwrite('识别结果.jpg', tem)
    cv2.imshow("output.jpg", tem)
    cv2.waitKey()


feature_matching()
