"""
简化版训练启动脚本
"""

import os
import sys
import subprocess
import torch

def check_environment():
    """检查训练环境"""
    print("=" * 60)
    print("中国跳棋神经网络训练环境检查")
    print("=" * 60)

    # 检查Python版本
    print(f"Python版本: {sys.version}")

    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU内存: {gpu_memory:.1f} GB")
        print(f"  GPU数量: {torch.cuda.device_count()}")
    else:
        print("✗ CUDA不可用，将使用CPU训练")
        print("  建议安装CUDA版本的PyTorch以获得更快训练速度")

    # 检查目录
    directories = ["models", "training_data", "logs"]
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  创建目录: {dir_name}")

    print("=" * 60)

def install_requirements():
    """安装必要的依赖"""
    print("安装依赖包...")
    requirements = [
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "tqdm",
        "pygame"
    ]

    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装，正在安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """主函数"""
    # 检查环境
    check_environment()

    # 安装依赖
    install_requirements()

    # 导入训练模块
    print("\n开始训练...")
    try:
        from train import main as train_main
        train_main()
    except ImportError as e:
        print(f"导入训练模块失败: {e}")
        print("请确保以下文件在相同目录:")
        print("  - model.py")
        print("  - self_play.py")
        print("  - train.py")
    except Exception as e:
        print(f"训练过程出错: {e}")

if __name__ == "__main__":
    main()