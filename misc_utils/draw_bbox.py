import cv2
import numpy as np

def project_3d_to_2d(bbox, K, extrinsic=None):
    """
    将已经在相机坐标系下的3D点投影到图像平面
    
    参数:
        bbox: (N, 3) 数组，表示在相机坐标系下的3D点集
        K: (3, 3) 相机内参矩阵
        extrinsic: 不再需要使用，保留参数是为了兼容之前的调用方式（可选）

    返回:
        points_2d: (N, 2) 投影后的2D像素坐标
        distances: 每个点到相机的距离
    """
    points_2d = []
    distances = []

    for point in bbox:
        # point 是已经在相机坐标系下的点，无需变换
        x, y, z = point
        distance = np.linalg.norm(point)  # 计算距离
        distances.append(distance)

        if z <= 0:
            # 如果点在相机后方，跳过（避免除以0或负值）
            points_2d.append(np.array([-1, -1]))  # 可标记为无效点
            continue

        # 使用内参K进行投影
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]

        points_2d.append(np.array([u, v]))

    # 转换为 numpy 数组并转为整数（用于 OpenCV 绘图）
    points_2d = np.array(points_2d, dtype=np.float32)
    return np.round(points_2d).astype(np.int32), distances
def adjust_color_by_distance(base_color, distance, max_distance=100):
    factor = 1 - min(distance / max_distance, 1)  # 根据距离计算颜色调整因子
    adjusted_color = tuple(min(int(c * (1 + factor)), 255) for c in base_color)  # 调整颜色深浅
    return adjusted_color

def draw_bbox(img_in, bbox, distances, base_color=(0, 0, 255)):
    '''
    img: 待画图的图像
    bbox: 待画图的bbox，8x2 (经过投影变换后应为2D点)
    distances: 对应的距离信息用于颜色调整
    
    注意：此函数假设bbox已经从3D投影到2D，并且是8个点表示的一个立方体的顶点。
    
    '''
    img = img_in.copy()
    height, width, _ = img.shape # 获取图像的高度和宽度


    def is_within_image(x, y, w, h):
        """判断点(x, y)是否在图像边界内"""
        return 0 <= x < w and 0 <= y < h
    
    # print(img.shape)
    # print(img.dtype)
    # print(img.flags['C_CONTIGUOUS'])

    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    
    for k in range(0, 4):
        color_bottom = adjust_color_by_distance(base_color, distances[k])
        if is_within_image(bbox[k][0], bbox[k][1], width, height) and is_within_image(bbox[(k+1) % 4][0], bbox[(k+1) % 4][1], width, height):
            img = cv2.line(img, tuple(bbox[k]), tuple(bbox[(k+1) % 4]), color_bottom, 2)
        
        color_top = adjust_color_by_distance(base_color, distances[k + 4])
        if is_within_image(bbox[k+4][0], bbox[k+4][1], width, height) and is_within_image(bbox[(k+1) % 4 + 4][0], bbox[(k+1) % 4 + 4][1], width, height):
            img = cv2.line(img, tuple(bbox[k+4]), tuple(bbox[(k+1) % 4 + 4]), color_top, 2)
        
        color_side = adjust_color_by_distance(base_color, (distances[k] + distances[k+4]) / 2)
        if is_within_image(bbox[k][0], bbox[k][1], width, height) and is_within_image(bbox[k+4][0], bbox[k+4][1], width, height):
            img = cv2.line(img, tuple(bbox[k]), tuple(bbox[k+4]), color_side, 2)
    
    return img

def render_bboxes(gt_pose, pose, bbox, K, img):
    # bbox_gt_transformed = np.array([np.dot(gt_pose[:3, :3], p) + gt_pose[:3, 3] for p in bbox])
    # bbox_transformed = np.array([np.dot(pose[:3, :3], p) + pose[:3, 3] for p in bbox])

    bbox_gt_transformed = (gt_pose[:3, :3] @ bbox.T).T + gt_pose[:3, 3]
    bbox_transformed = (pose[:3, :3] @ bbox.T).T + pose[:3, 3]

    bbox_gt_2d, distances_gt = project_3d_to_2d(bbox_gt_transformed, K, gt_pose)
    bbox_2d, distances_pred = project_3d_to_2d(bbox_transformed, K, pose)

    img1 = draw_bbox(img, bbox_gt_2d, distances_gt, (0, 0, 255))  # gt_bbox用红色基色
    img2 = draw_bbox(img1, bbox_2d, distances_pred, (255, 0, 0))  # 另一个bbox用蓝色基色
    
    return img2