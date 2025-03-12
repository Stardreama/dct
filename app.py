from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from inference import FaceForensicsPredictor

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 初始化预测器
predictor = FaceForensicsPredictor()

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html', test_results=get_test_results())

def get_test_results():
    """获取测试结果统计信息"""
    # 从CSV文件中读取测试结果
    import pandas as pd
    try:
        df = pd.read_csv('./test_results/forgery_type_comparison.csv')
        return {
            'overall_accuracy': 0.9218,  # 从您的测试结果中获取
            'overall_auc': 0.9783,
            'forgery_types': df.to_dict('records')
        }
    except:
        return {
            'overall_accuracy': 0.9218,
            'overall_auc': 0.9783,
            'forgery_types': []
        }

@app.route('/predict', methods=['POST'])
def predict():
    """处理图像预测请求"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件上传'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'})
    
    if file and allowed_file(file.filename):
        try:
            # 生成唯一的文件名
            filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # 记录开始时间
            start_time = datetime.now()
            
            # 预测
            result = predictor.predict(file_path)
            
            # 计算耗时
            inference_time = (datetime.now() - start_time).total_seconds()
            
            if result['success']:
                # 生成结果图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"result_{timestamp}_{os.path.splitext(filename)[0]}.png"
                result_path = os.path.join(RESULT_FOLDER, result_filename)
                
                predictor.generate_visualization(file_path, result, result_path)
                
                # 返回结果
                return jsonify({
                    'success': True,
                    'prediction': result['prediction'],
                    'is_fake': result['is_fake'],
                    'real_probability': result['real_probability'],
                    'fake_probability': result['fake_probability'],
                    'original_image': url_for('static', filename=f'uploads/{filename}'),
                    'result_image': url_for('static', filename=f'results/{result_filename}'),
                    'inference_time': f"{inference_time:.2f}秒"
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result['error']
                })
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f"处理过程中出错: {str(e)}"
            })
    
    return jsonify({'success': False, 'error': '不支持的文件类型'})

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)