import torch

def inspect_pth_file(pth_path):
    """查看.pth文件的所有键名和形状"""
    print(f"检查文件: {pth_path}")
    
    # 加载权重文件
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # 检查文件类型
    if isinstance(checkpoint, dict):
        print(f"文件类型: 字典 (包含 {len(checkpoint)} 个键)")
        
        # 如果是state_dict或model
        if 'state_dict' in checkpoint:
            print("找到 'state_dict' 键")
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            print("找到 'model' 键")  
            state_dict = checkpoint['model']
        else:
            print("使用直接state_dict")
            state_dict = checkpoint
        
        # 显示所有键名
        print(f"\n所有键名 ({len(state_dict)} 个):")
        for i, key in enumerate(state_dict.keys()):
            value = state_dict[key]
            shape = value.shape if hasattr(value, 'shape') else 'N/A'
            print(f"{i:4d}: {key:80} {shape}")
            
    else:
        print(f"文件类型: {type(checkpoint)}")
        print("这不是一个字典格式的权重文件")

# 使用
inspect_pth_file("/home/wangxiao/6d_pose_learn/src/myproject/gspose_model/model_weights.pth")