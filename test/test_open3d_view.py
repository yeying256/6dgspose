import open3d as o3d
import numpy as np

# 定义一个齐次变换矩阵（这里仅做平移）
transform_matrix = np.array([
    [1, 0, 0, 1],  # 沿 x 轴平移 1 单位
    [0, 1, 0, 2],  # 沿 y 轴平移 2 单位
    [0, 0, 1, 3],  # 沿 z 轴平移 3 单位
    [0, 0, 0, 1]
])

# 创建原始坐标系
original_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

# 创建变换后的坐标系
transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
transformed_frame.transform(transform_matrix)

# 可视化
o3d.visualization.draw_geometries([original_frame, transformed_frame])