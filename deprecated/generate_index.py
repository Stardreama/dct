import os
import glob
import random
from tqdm import tqdm

def generate_index_files(dataset_dir):
    """为训练、验证和测试集生成索引文件"""
    
    for split in ["train", "valid", "test"]:
        file_list = []
        print(f"生成 {split}.txt 索引文件...")
        
        # 处理真实图像 (标签0)
        real_dir = os.path.join(dataset_dir, split, "real")
        if os.path.exists(real_dir):
            real_subfolders = [os.path.join(real_dir, d) for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))]
            
            for subfolder in tqdm(real_subfolders, desc=f"处理{split}集真实图像"):
                frame_files = glob.glob(os.path.join(subfolder, "*.jpg"))
                for frame_path in frame_files:
                    # 使用绝对路径
                    abs_path = os.path.abspath(frame_path)
                    file_list.append((abs_path, 0))
        
        # 处理伪造图像 (标签1)
        for fake_type in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            fake_dir = os.path.join(dataset_dir, split, "fake", fake_type)
            if os.path.exists(fake_dir):
                fake_subfolders = [os.path.join(fake_dir, d) for d in os.listdir(fake_dir) if os.path.isdir(os.path.join(fake_dir, d))]
                
                for subfolder in tqdm(fake_subfolders, desc=f"处理{split}集{fake_type}图像"):
                    frame_files = glob.glob(os.path.join(subfolder, "*.jpg"))
                    for frame_path in frame_files:
                        # 使用绝对路径
                        abs_path = os.path.abspath(frame_path)
                        file_list.append((abs_path, 1))
        
        # 打乱文件列表
        random.seed(42)
        random.shuffle(file_list)
        
        # 写入索引文件
        out_file = os.path.join(dataset_dir, f"{split}.txt")
        with open(out_file, "w") as f:
            for file_path, label in file_list:
                f.write(f"{file_path} {label}\n")
        
        print(f"{split}.txt 创建完成，包含 {len(file_list)} 条记录")
        
        # 统计数据分布
        total = len(file_list)
        real_count = sum(1 for _, label in file_list if label == 0)
        fake_count = sum(1 for _, label in file_list if label == 1)
        
        print(f"数据集统计 ({os.path.basename(out_file)}):")
        print(f"  - 总计: {total}张图像")
        if total > 0:
            print(f"  - 真实: {real_count}张 ({real_count/total*100:.1f}%)")
            print(f"  - 伪造: {fake_count}张 ({fake_count/total*100:.1f}%)")

if __name__ == "__main__":
    dataset_dir = "D:/project/DCT_RGB_HRNet/dataset"
    generate_index_files(dataset_dir)