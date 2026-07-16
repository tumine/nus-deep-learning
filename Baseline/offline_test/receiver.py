from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# 保存图片的目录（可修改）
SAVE_DIR = r"C:\Documents\nus-deep-learning\Baseline\img_recv"


# 确保目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    """接收树莓派发来的图片并保存"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({"status": "error", "msg": "No image file"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "msg": "Empty filename"}), 400

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        # 保存文件
        file.save(save_path)
        print(f"✅ 收到图片并保存: {save_path}")

        return jsonify({"status": "success", "msg": f"Saved as {filename}"}), 200

    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == '__main__':
    # 监听所有网卡，端口可修改（例如 5001）
    app.run(host='0.0.0.0', port=5001, debug=False)