import os
import glob

def verify_dataset_structure(dataset_dir):
    """验证数据集目录结构"""
    print(f"验证数据集结构: {dataset_dir}")
    
    all_ok = True
    
    # 检查主要目录
    for split in ["train", "valid", "test"]:
        # 检查真实图像
        real_dir = os.path.join(dataset_dir, split, "real")
        if not os.path.exists(real_dir):
            print(f"错误: {real_dir} 不存在")
            all_ok = False
        else:
            real_subfolders = [d for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))]
            real_images = sum([len(glob.glob(os.path.join(real_dir, d, "*.jpg"))) for d in real_subfolders])
            print(f"{split}/real: {len(real_subfolders)}个文件夹, {real_images}张图像")
            
            if real_images == 0:
                print(f"警告: {split}/real 中没有找到图像")
                all_ok = False
        
        # 检查伪造图像
        for fake_type in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            fake_dir = os.path.join(dataset_dir, split, "fake", fake_type)
            if not os.path.exists(fake_dir):
                print(f"错误: {fake_dir} 不存在")
                all_ok = False
            else:
                fake_subfolders = [d for d in os.listdir(fake_dir) if os.path.isdir(os.path.join(fake_dir, d))]
                fake_images = sum([len(glob.glob(os.path.join(fake_dir, d, "*.jpg"))) for d in fake_subfolders])
                print(f"{split}/fake/{fake_type}: {len(fake_subfolders)}个文件夹, {fake_images}张图像")
                
                if fake_images == 0:
                    print(f"警告: {split}/fake/{fake_type} 中没有找到图像")
                    all_ok = False
    
    # 检查掩码
    for fake_type in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        mask_dir = os.path.join(dataset_dir, "mask", fake_type)
        if not os.path.exists(mask_dir):
            print(f"错误: {mask_dir} 不存在")
            all_ok = False
        else:
            mask_subfolders = [d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]
            mask_images = sum([len(glob.glob(os.path.join(mask_dir, d, "*.jpg"))) for d in mask_subfolders])
            print(f"mask/{fake_type}: {len(mask_subfolders)}个文件夹, {mask_images}张图像")
            
            if mask_images == 0:
                print(f"警告: mask/{fake_type} 中没有找到图像")
                all_ok = False
    
    if all_ok:
        print("\n验证成功: 所有目录都包含数据")
    else:
        print("\n验证失败: 部分目录缺少数据")
    
    return all_ok

if __name__ == "__main__":
    dataset_dir = "D:/project/DCT_RGB_HRNet/dataset"
    verify_dataset_structure(dataset_dir)