"""
计算模板匹配结果中心坐标,根据左上角点和宽高求出目标区域.
"""
import cv2


def get_target_rectangle(left_top_pos, w, h):
    x_min, y_min = left_top_pos
    # 中心位置的坐标:
    x_middle, y_middle = int(x_min + w / 2), int(y_min + h / 2)
    # 左下(min,max)->右下(max,max)->右上(max,min)
    left_bottom_pos, right_bottom_pos = (x_min, y_min + h), (x_min + w, y_min + h)
    right_top_pos = (x_min + w, y_min)
    # 点击位置:
    middle_point = [x_middle, y_middle]
    # 识别目标区域: 点序:左上->左下->右下->右上, 左上(min,min)右下(max,max)
    rectangle = (left_top_pos, left_bottom_pos, right_bottom_pos, right_top_pos)
    return middle_point, rectangle


"""
模板匹配
"""


def template_matching(templatePicPath, targetPicPath):
    template = cv2.imread(templatePicPath)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    targetPic = cv2.imread(targetPicPath, 0)
    """
    result保存结果的矩阵，我们可以通过minMaxLoc() 确定结果矩阵的最大值和最小值的位置
    &minVal 和 &maxVal: 在矩阵 result 中存储的最小值和最大值
    &minLoc 和 &maxLoc: 在结果矩阵中最小值和最大值的坐标.
    """
    result = cv2.matchTemplate(gray, targetPic, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    height, width = targetPic.shape[:2]
    top_left = max_loc
    middle_point, rectangle = get_target_rectangle(top_left, height, width)
    print(middle_point)
    return middle_point


"""
特征点匹配并返回匹配结果坐标
"""


def feature_matching(templatePicPath, targetPicPath):
    tem = cv2.imread(templatePicPath)
    tar = cv2.imread(targetPicPath)

    # 创建特征检测器——用于检测模板和图像上的获取图像特征的关键点和描述符
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tem, None)
    kp2, des2 = sift.detectAndCompute(tar, None)
    """
    获取图片的长，宽
    img.shape[:2]
    取彩色图片的长、宽
    img.shape[:3]
    取彩色图片的长、宽、通道
    img.shape[0]
    图像的垂直尺寸（高度）
    img.shape[1]
    图像的水平尺寸（宽度）
    img.shape[2]
    图像的通道数
    """
    height1, width2 = tem.shape[:2]
    print(height1, width2)

    """
    定义FLANN匹配器
    indexParams:配置我们要使用的算法Randomized k-d tree,增加树的数量能加快搜索速度，但由于内存负载的问题，树的数量只能控制在一定范围内，比如20，如果超过一定范围，那么搜索速度不会增加甚至会减慢
    SearchParams：指定递归遍历的次数,值越高结果越准确，但是消耗的时间也越多
    """
    indexParams = dict(algorithm=0, trees=10)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    # 使用KNN算法实现图像匹配，返回最佳的k个匹配,KnnMatch与match的返回值类型一样，只不过一组返回的俩个DMatch类型：
    matches = flann.knnMatch(des1, des2, k=2)
    # matches是DMatch对象，具有以下属性：
    # DMatch.distance - 描述符之间的距离。 越低越好。
    # DMatch.trainIdx - 训练描述符中描述符的索引
    # DMatch.queryIdx - 查询描述符中描述符的索引
    # DMatch.imgIdx - 训练图像的索引。
    """
    matches是DMatch对象，具有以下属性：
    DMatch.distance - 描述符之间的距离。 越低越好。
    DMatch.trainIdx - 训练描述符中描述符的索引
    DMatch.queryIdx - 查询描述符中描述符的索引
    DMatch.imgIdx - 训练图像的索引。
    """

    """
    去除错误匹配，0.5是系数，系数大小不同，匹配的结果页不同
    distance:代表匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近
    """
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.25 * n.distance:
            goodMatches.append(m)

    # 获取某个点的坐标位置
    # index是获取匹配结果的中位数
    index = int(len(goodMatches) / 2)
    # queryIdx是目标图像的描述符索引
    x, y = kp1[goodMatches[index].queryIdx].pt
    # 将识别结果绘制在模板上并保存
    cv2.rectangle(tem, (int(x), int(y)), (int(x) + 30, int(y) + 30), (0, 0, 255), 5)
    cv2.imwrite('识别结果.jpg', tem)
    return [int(x), int(y)]


def response_format(errorCode, errorMessage, pos):
    res = {"errorCode": errorCode, "errorMessage": errorMessage, "pos": pos}
    return str(res)
