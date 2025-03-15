import os
import json
import cv2
from tqdm import tqdm
from PIL import Image
import shutil
import glob

def reorganize_dataset(source_dir, target_dir):
    """特别为视频对格式的JSON文件设计的数据集重组函数"""
    # 创建目标目录结构
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(target_dir, split, "real"), exist_ok=True)
        for fake_type in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            os.makedirs(os.path.join(target_dir, split, "fake", fake_type), exist_ok=True)
            os.makedirs(os.path.join(target_dir, "mask", fake_type), exist_ok=True)
    
    # 读取JSON文件
    json_files = {
        "train": os.path.join(source_dir, "train.json"),
        "valid": os.path.join(source_dir, "val.json"),
        "test": os.path.join(source_dir, "test.json")
    }
    
    # 首先处理真实视频
    process_real_videos(source_dir, target_dir)
    
    # 然后处理伪造视频对
    for split, json_file in json_files.items():
        if not os.path.exists(json_file):
            print(f"警告: {json_file} 不存在")
            continue
            
        with open(json_file, "r") as f:
            pairs = json.load(f)
        
        print(f"处理 {split} 集中的 {len(pairs)} 个视频对...")
        
        # 处理每个视频对
        for pair in tqdm(pairs, desc=f"处理{split}视频对"):
            if not isinstance(pair, list) or len(pair) < 2:
                print(f"跳过无效的视频对: {pair}")
                continue
                
            source_id, target_id = pair[0], pair[1]
            video_pair_id = f"{source_id}_{target_id}"
            
            # 对每种伪造类型处理视频对
            for fake_type in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
                # 复制伪造视频帧
                src_frames_path = os.path.join(source_dir, "manipulated_sequences", fake_type, "c23", "frames", video_pair_id)
                if os.path.exists(src_frames_path):
                    dst_frames_path = os.path.join(target_dir, split, "fake", fake_type, video_pair_id)
                    copy_frames_to_jpg(src_frames_path, dst_frames_path)
                    
                    # 复制对应的掩码
                    src_masks_path = os.path.join(source_dir, "manipulated_sequences", fake_type, "c23", "masks", video_pair_id)
                    if os.path.exists(src_masks_path):
                        dst_masks_path = os.path.join(target_dir, "mask", fake_type, video_pair_id)
                        copy_frames_to_jpg(src_masks_path, dst_masks_path, is_mask=True)
    
    print("数据集重组完成！")

def process_real_videos(source_dir, target_dir):
    """处理真实视频，将它们分配到训练、验证和测试集"""
    print("处理真实视频...")
    
    # 查找所有真实视频
    youtube_dir = os.path.join(source_dir, "original_sequences", "youtube", "c23", "frames")
    if not os.path.exists(youtube_dir):
        print(f"错误: 未找到真实视频目录 {youtube_dir}")
        return
        
    # 获取所有视频ID
    all_video_ids = [d for d in os.listdir(youtube_dir) if os.path.isdir(os.path.join(youtube_dir, d))]
    print(f"找到 {len(all_video_ids)} 个真实视频目录")
    
    # 读取JSON文件，确定哪些视频ID被用作测试和验证集
    used_in_train = set()
    used_in_valid = set()
    used_in_test = set()
    
    # 从train.json收集
    train_json = os.path.join(source_dir, "train.json")
    if os.path.exists(train_json):
        with open(train_json, "r") as f:
            for pair in json.load(f):
                if isinstance(pair, list) and len(pair) >= 2:
                    used_in_train.add(pair[0])
                    used_in_train.add(pair[1])
    
    # 从val.json收集
    val_json = os.path.join(source_dir, "val.json")
    if os.path.exists(val_json):
        with open(val_json, "r") as f:
            for pair in json.load(f):
                if isinstance(pair, list) and len(pair) >= 2:
                    used_in_valid.add(pair[0])
                    used_in_valid.add(pair[1])
    
    # 从test.json收集
    test_json = os.path.join(source_dir, "test.json")
    if os.path.exists(test_json):
        with open(test_json, "r") as f:
            for pair in json.load(f):
                if isinstance(pair, list) and len(pair) >= 2:
                    used_in_test.add(pair[0])
                    used_in_test.add(pair[1])
    
    # 复制视频到相应的集合
    for video_id in tqdm(all_video_ids, desc="复制真实视频"):
        src_path = os.path.join(youtube_dir, video_id)
        
        if video_id in used_in_test:
            dst_path = os.path.join(target_dir, "test", "real", video_id)
            copy_frames_to_jpg(src_path, dst_path)
        elif video_id in used_in_valid:
            dst_path = os.path.join(target_dir, "valid", "real", video_id)
            copy_frames_to_jpg(src_path, dst_path)
        else:
            dst_path = os.path.join(target_dir, "train", "real", video_id)
            copy_frames_to_jpg(src_path, dst_path)

def copy_frames_to_jpg(src_dir, dst_dir, is_mask=False, sample_frames=None):
    """将PNG帧转换为JPG并复制到目标目录"""
    if not os.path.exists(src_dir):
        return
    
    os.makedirs(dst_dir, exist_ok=True)
    
    # 获取所有PNG文件
    frame_files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    
    # 如果需要抽样，只选择部分帧
    if sample_frames and len(frame_files) > sample_frames:
        import random
        random.seed(42)  # 固定随机种子以确保可重复性
        frame_files = random.sample(frame_files, sample_frames)
    
    for png_file in frame_files:
        try:
            src_file = os.path.join(src_dir, png_file)
            
            # 从文件名中提取帧号 (例如: 000.png -> 0)
            frame_num = int(os.path.splitext(png_file)[0])
            dst_file = os.path.join(dst_dir, f"frame{frame_num}.jpg")
            
            if is_mask:
                # 如果是掩码图像，使用PIL处理以确保转换为灰度
                img = Image.open(src_file).convert("L")
                img.save(dst_file)
            else:
                # 普通图像，直接用OpenCV转换
                img = cv2.imread(src_file)
                if img is None:
                    print(f"警告: 无法读取图像 {src_file}")
                    continue
                cv2.imwrite(dst_file, img)
        except Exception as e:
            print(f"处理文件 {png_file} 时出错: {e}")

def check_frame_counts(target_dir):
    """检查重组后的数据集中图像数量"""
    counts = {}
    for root, dirs, files in os.walk(target_dir):
        jpg_count = len([f for f in files if f.endswith('.jpg')])
        if jpg_count > 0:
            rel_path = os.path.relpath(root, target_dir)
            counts[rel_path] = jpg_count
    
    print("\n数据集图像统计:")
    for path, count in sorted(counts.items()):
        print(f"{path}: {count}张图像")

if __name__ == "__main__":
    source_dir = "D:/document/FaceForensics++"  # 原始数据集路径
    target_dir = "D:/project/DCT_RGB_HRNet/dataset"  # 目标数据集路径
    
    # 清空目标目录以确保干净重建（可选）
    if os.path.exists(target_dir):
        print(f"清空目标目录: {target_dir}")
        shutil.rmtree(target_dir)
    
    # 重组数据集
    reorganize_dataset(source_dir, target_dir)
    
    # 检查结果
    check_frame_counts(target_dir)