import sys
import os
# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image
import yaml
import easydict
from trainer import Trainer  # 现在可以正常导入了
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 不使用GUI后端
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

class FaceForensicsPredictor:
    def __init__(self, model_path='./checkpoints/FAD_RGB_F2Fc0/best.pkl', config_path='./config.yaml'):
        """初始化模型预测类"""
        print("初始化伪造检测模型...")
        
        # 设置中文字体
        self.setup_chinese_font()
        
        # 加载配置
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        self.config = easydict.EasyDict(config)
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # 加载模型
        pretrained_path = './pretrained/xception-b5690688.pth'
        self.model = Trainer(self.config, [0] if torch.cuda.is_available() else [], 'FAD', pretrained_path)
        
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"模型已加载: {model_path}")
    
    def setup_chinese_font(self):
        """配置中文字体支持"""
        # 尝试几种常见的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        font_found = False
        
        for font_name in chinese_fonts:
            try:
                font_path = fm.findfont(FontProperties(family=font_name))
                if font_path and not font_path.endswith('DejaVuSans.ttf'):  # 避免使用后备字体
                    plt.rcParams['font.family'] = [font_name, 'sans-serif']
                    print(f"使用中文字体: {font_name}")
                    font_found = True
                    break
            except:
                continue
        
        if not font_found:
            # 如果找不到合适的中文字体，尝试使用系统默认字体
            print("警告：未找到合适的中文字体，将使用系统默认字体")
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei']
        
        # 确保可以显示负号
        plt.rcParams['axes.unicode_minus'] = False

    def predict(self, image_path):
        """预测图像是否为伪造"""
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # 保存原始尺寸
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                print(f"模型输出类型: {type(outputs)}")
                
                # 检查是否有掩码输出
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    mask, outputs = outputs
                    print("模型返回了掩码数据")
                else:
                    mask = None
                    print("模型没有返回掩码数据")
                    
                probs = torch.nn.Sigmoid()(outputs)
                _, predicted = torch.max(outputs.data, 1)
            
            # 处理结果
            prediction = "FAKE" if predicted.item() == 1 else "REAL"
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()
            
            # 修改后的掩码处理
            mask_info = None
            if mask is not None:
                try:
                    print("开始处理掩码数据...")
                    mask_np = mask.cpu().numpy()
                    print(f"掩码数组形状: {mask_np.shape}")
                    
                    # 确保掩码是2D的
                    if len(mask_np.shape) > 2:
                        # 如果有多个通道，取第一个
                        mask_np = mask_np[0, 0]
                    
                    # 使用增强函数处理掩码
                    mask_normalized = self.enhance_mask(mask_np)
                    
                    # 创建结果信息
                    mask_info = {
                        'data': mask_normalized,
                        'shape': mask_normalized.shape,
                        'original_image_size': original_size
                    }
                    print(f"掩码处理完成，形状: {mask_normalized.shape}")
                except Exception as e:
                    print(f"处理掩码时出错: {e}")
                    import traceback
                    print(traceback.format_exc())
            else:
                print("没有掩码数据可处理")
                
            print(f"原始图像尺寸: {original_size}")
            
            return {
                'success': True,
                'prediction': prediction,
                'is_fake': predicted.item() == 1,
                'real_probability': real_prob,
                'fake_probability': fake_prob,
                'mask': mask_info['data'] if mask_info else None
            }
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }

    def generate_visualization(self, image_path, result, output_path):
        """生成结果可视化（改进掩码对齐）"""
        self.setup_chinese_font()  # 使用类方法
        
        try:
            # 加载原始图像
            original_image = Image.open(image_path).convert('RGB')
            original_np = np.array(original_image)
            orig_h, orig_w = original_np.shape[:2]
            
            plt.figure(figsize=(15, 5))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("原始图像", fontsize=14)
            plt.axis('off')
            
            # 掩码图像
            plt.subplot(1, 3, 2)
            if result.get('mask') is not None:
                mask = result['mask']
                
                # 确保掩码是2D数组
                if isinstance(mask, dict):
                    # 如果我们得到了一个字典而不是直接的数组
                    mask = mask.get('data', mask)
                
                # 获取掩码的原始尺寸并调整大小以匹配原始图像
                mask_resized = cv2.resize(mask, (orig_w, orig_h), 
                                         interpolation=cv2.INTER_LINEAR)
                
                # 显示调整后的掩码
                plt.imshow(mask_resized, cmap='jet')
                plt.title("检测掩码", fontsize=14)
                
                # 为第三个子图准备融合图像
                heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), 
                                           cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(original_np, 0.7, heatmap, 0.3, 0)
            else:
                plt.title("无掩码数据", fontsize=14)
                overlay = original_np
            plt.axis('off')
            
            # 融合图像
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title(f"判断结果: {'伪造图像' if result['is_fake'] else '真实图像'}\n"
                     f"真实概率: {result['real_probability']:.4f}, "
                     f"伪造概率: {result['fake_probability']:.4f}", 
                     fontsize=14)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"可视化结果已保存至: {output_path}")
            return True
            
        except Exception as e:
            print(f"生成可视化结果时出错: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def enhance_mask(self, mask_np):
        """增强掩码对比度，即使值范围很小也能产生可见差异"""
        # 计算范围
        mask_min = mask_np.min()
        mask_max = mask_np.max()
        print(f"原始掩码值范围: {mask_min:.6f} 到 {mask_max:.6f}, 差值: {mask_max-mask_min:.6f}")
        
        # 如果范围太小，使用更激进的标准化方法
        if mask_max - mask_min < 0.1:
            print("掩码值范围过小，应用强化对比度提升")
            # 方法1: 尝试使用均值和标准差标准化
            mean = mask_np.mean()
            std = mask_np.std()
            print(f"掩码均值: {mean:.6f}, 标准差: {std:.6f}")
            
            if std > 0.001:  # 确保标准差不是太小
                mask_normalized = (mask_np - mean) / (std + 1e-8)
                # 再次线性拉伸到0-1范围
                new_min = mask_normalized.min()
                new_max = mask_normalized.max()
                mask_normalized = (mask_normalized - new_min) / (new_max - new_min + 1e-8)
            else:
                # 方法2: 如果标准差也很小，尝试取绝对值后增强
                print("掩码标准差很小，尝试取绝对值并增强...")
                mask_abs = np.abs(mask_np)
                # 找到前10%和后10%的阈值
                sorted_values = np.sort(mask_abs.flatten())
                low_threshold = sorted_values[int(len(sorted_values) * 0.1)]
                high_threshold = sorted_values[int(len(sorted_values) * 0.9)]
                
                # 通过阈值截断增强对比度
                mask_normalized = np.clip(mask_abs, low_threshold, high_threshold)
                mask_normalized = (mask_normalized - low_threshold) / (high_threshold - low_threshold + 1e-8)
        else:
            # 常规的min-max归一化
            mask_normalized = (mask_np - mask_min) / (mask_max - mask_min)
        
        print(f"增强后的掩码值范围: {mask_normalized.min():.6f} 到 {mask_normalized.max():.6f}")
        return mask_normalized
    
# 测试代码
if __name__ == "__main__":
    predictor = FaceForensicsPredictor()
    # 假设有一个测试图像
    test_image = "path_to_test_image.jpg"
    if os.path.exists(test_image):
        result = predictor.predict(test_image)
        if result["success"]:
            print(f"预测结果: {result['prediction']}")
            print(f"真实概率: {result['real_probability']:.4f}")
            print(f"伪造概率: {result['fake_probability']:.4f}")
            
            # 生成可视化
            output_path = "test_result.png"
            predictor.generate_visualization(test_image, result, output_path)
            print(f"可视化结果已保存至: {output_path}")