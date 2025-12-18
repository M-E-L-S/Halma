"""
完整的神经网络训练脚本
支持GPU加速和真实数据训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import sys
import time
import pickle
import math
from datetime import datetime
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from src.core.board import ChineseCheckersBoard

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 检查GPU
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("使用CPU训练")

# 导入自定义模块
try:
    from model import CheckersNet, BoardEncoder, NeuralEvaluator
    from self_play import RealSelfPlayGenerator, TrainingSample
    print("模块导入成功")
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


class CheckersDataset(Dataset):
    """中国跳棋数据集"""

    def __init__(self, samples: List[TrainingSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 转换为张量
        features = torch.FloatTensor(sample.features)
        value_target = torch.FloatTensor([sample.value_target])

        if self.transform:
            features = self.transform(features)

        return features, value_target


class CheckersTrainer:
    """中国跳棋神经网络训练器"""

    def __init__(self,
                 model_path: str = "models/checkers_net.pth",
                 data_path: str = "data/training_data.pkl",
                 log_dir: str = "logs",
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        """
        初始化训练器
        """
        self.model_path = model_path
        self.data_path = data_path
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        # 创建目录
        self._create_directories()

        # 初始化编码器
        self.encoder = BoardEncoder()

        # 初始化模型
        self.model = CheckersNet(
            input_channels=11,
            board_size=self.encoder.get_board_size()
        ).to(self.device)

        # 加载现有模型
        self._load_model()

        # 优化器和损失函数
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }

        # 自对弈生成器
        self.generator = RealSelfPlayGenerator()

        # 日志文件
        self.log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self._setup_logging()

        print(f"\n训练器初始化完成:")
        print(f"  设备: {self.device}")
        print(f"  模型路径: {model_path}")
        print(f"  数据路径: {data_path}")
        print(f"  日志目录: {log_dir}")
        print(f"  批大小: {batch_size}")
        print(f"  学习率: {learning_rate}")

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            os.path.dirname(self.model_path),
            os.path.dirname(self.data_path),
            self.log_dir,
            "plots",
            "checkpoints"
        ]

        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")

    def _setup_logging(self):
        """设置日志"""
        with open(self.log_file, 'w') as f:
            f.write(f"训练日志 - 开始时间: {datetime.now()}\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"设备: {self.device}\n\n")

    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        print(log_message)

        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

    def _load_model(self):
        """加载现有模型"""
        if os.path.exists(self.model_path):
            try:
                if self.device.type == 'cuda':
                    self.model.load_state_dict(torch.load(self.model_path))
                else:
                    self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))

                self._log(f"加载现有模型: {self.model_path}")
            except Exception as e:
                self._log(f"加载模型失败: {e}")
        else:
            self._log("初始化新模型")

    def _save_model(self, epoch: int = None):
        """保存模型"""
        if epoch is not None:
            # 保存检查点
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)

        # 保存最新模型
        torch.save(self.model.state_dict(), self.model_path)
        self._log(f"模型已保存: {self.model_path}")

    def load_training_data(self) -> List[TrainingSample]:
        """加载训练数据"""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)

                samples = data.get('samples', [])
                self._log(f"加载训练数据: {self.data_path} ({len(samples)} 个样本)")
                return samples
            except Exception as e:
                self._log(f"加载训练数据失败: {e}")

        return []

    def save_training_data(self, samples: List[TrainingSample]):
        """保存训练数据"""
        save_data = {
            'samples': samples,
            'timestamp': time.time(),
            'encoder_info': {
                'board_size': self.encoder.get_board_size()
            }
        }

        with open(self.data_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._log(f"训练数据已保存: {self.data_path} ({len(samples)} 个样本)")

    def prepare_dataloaders(self, samples: List[TrainingSample],
                           val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """准备数据加载器"""
        if not samples:
            raise ValueError("没有训练样本")

        # 划分训练集和验证集
        num_samples = len(samples)
        num_val = int(num_samples * val_ratio)
        num_train = num_samples - num_val

        # 随机打乱
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]

        # 创建数据集
        train_dataset = CheckersDataset([samples[i] for i in train_indices])
        val_dataset = CheckersDataset([samples[i] for i in val_indices])

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda')
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.batch_size * 2, len(val_dataset)),
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda')
        )

        self._log(f"数据划分: 训练集 {num_train} 样本, 验证集 {num_val} 样本")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for batch_idx, (features, targets) in enumerate(train_loader):
            # 移动到设备
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 前向传播
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)

            # 计算MAE
            mae = torch.abs(outputs - targets).mean().item()

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 更新统计
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1

            # 每10个batch打印一次进度
            if (batch_idx + 1) % 10 == 0:
                self._log(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}: "
                         f"Loss={loss.item():.4f}, MAE={mae:.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0

        return avg_loss, avg_mae

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean().item()

                total_loss += loss.item()
                total_mae += mae
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0

        return avg_loss, avg_mae

    def train(self, num_epochs: int = 50, val_ratio: float = 0.2):
        """训练模型"""
        # 加载训练数据
        samples = self.load_training_data()

        if not samples:
            self._log("错误: 没有训练数据")
            return

        # 准备数据加载器
        train_loader, val_loader = self.prepare_dataloaders(samples, val_ratio)

        self._log(f"\n开始训练:")
        self._log(f"  总epoch数: {num_epochs}")
        self._log(f"  训练样本: {len(train_loader.dataset)}")
        self._log(f"  验证样本: {len(val_loader.dataset)}")

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # 训练一个epoch
            train_loss, train_mae = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_mae = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rate'].append(current_lr)

            epoch_time = time.time() - epoch_start_time

            # 打印epoch结果
            self._log(f"Epoch {epoch}/{num_epochs}: "
                     f"Train Loss={train_loss:.4f}, Train MAE={train_mae:.4f}, "
                     f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}, "
                     f"LR={current_lr:.6f}, Time={epoch_time:.1f}s")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(epoch)
                patience_counter = 0
                self._log(f"  保存最佳模型 (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= max_patience:
                self._log(f"早停: 验证损失连续 {max_patience} 个epoch没有改善")
                break

            # 每5个epoch保存一次训练曲线
            if epoch % 5 == 0 or epoch == num_epochs:
                self.plot_training_history(epoch)

        self._log(f"\n训练完成，最佳验证损失: {best_val_loss:.4f}")

    def generate_data_and_train(self,
                               iterations: int = 5,
                               games_per_iteration: int = 20,
                               epochs_per_iteration: int = 10):
        """生成数据并训练（迭代过程）"""
        self._log(f"\n开始生成-训练迭代过程:")
        self._log(f"  迭代次数: {iterations}")
        self._log(f"  每轮游戏数: {games_per_iteration}")
        self._log(f"  每轮训练epoch数: {epochs_per_iteration}")

        for iteration in range(iterations):
            self._log(f"\n{'='*60}")
            self._log(f"迭代 {iteration + 1}/{iterations}")
            self._log(f"{'='*60}")

            # 1. 生成自对弈数据
            self._log("生成自对弈数据...")

            # 设置生成器参数
            self.generator.games_per_generation = games_per_iteration
            self.generator.neural_evaluator = NeuralEvaluator(
                model_path=self.model_path if os.path.exists(self.model_path) else None
            )

            # 生成数据
            start_time = time.time()
            new_samples = self.generator.generate_batch(
                batch_size=games_per_iteration * 15  # 估计值
            )
            gen_time = time.time() - start_time

            if not new_samples:
                self._log("警告: 没有生成新数据")
                continue

            self._log(f"生成 {len(new_samples)} 个新样本，耗时 {gen_time:.1f}秒")

            # 2. 加载现有数据
            existing_samples = self.load_training_data()

            # 3. 合并数据（限制总样本数）
            all_samples = existing_samples + new_samples
            max_samples = 10000  # 最大样本数

            if len(all_samples) > max_samples:
                # 随机抽样保留最大样本数
                import random
                all_samples = random.sample(all_samples, max_samples)
                self._log(f"数据量过多，随机抽样保留 {max_samples} 个样本")

            # 4. 保存数据
            self.save_training_data(all_samples)

            # 5. 训练模型
            self._log(f"开始训练 (使用 {len(all_samples)} 个样本)...")
            self.train(num_epochs=epochs_per_iteration)

            # 6. 测试模型
            self.test_model_performance()

        self._log(f"\n生成-训练迭代过程完成")

    def test_model_performance(self):
        """测试模型性能"""
        self._log("\n测试模型性能...")

        # 创建测试局面
        board = ChineseCheckersBoard()

        # 测试不同局面的评估
        test_cases = [
            ("初始局面", board.cells.copy(), 1),
            ("初始局面", board.cells.copy(), -1),
        ]

        self.model.eval()

        for name, board_state, player in test_cases:
            try:
                # 编码棋盘
                features = self.encoder.encode_board(board_state, player)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

                # 推理
                with torch.no_grad():
                    output = self.model(features_tensor)
                    score = output.item()

                self._log(f"  {name} (玩家{player}): {score:.4f}")

            except Exception as e:
                self._log(f"  测试失败: {e}")

    def plot_training_history(self, epoch: int):
        """绘制训练历史曲线"""
        if not self.history['train_loss']:
            return

        plt.figure(figsize=(15, 5))

        # 1. 损失曲线
        plt.subplot(1, 3, 1)
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. MAE曲线
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['train_mae'], 'b-', label='Train MAE', linewidth=2, alpha=0.7)
        plt.plot(epochs, self.history['val_mae'], 'r-', label='Val MAE', linewidth=2, alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. 学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # 保存图片
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"plots/training_history_epoch_{epoch}_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        self._log(f"训练曲线已保存: {plot_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("中国跳棋神经网络训练系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 配置参数
    config = {
        'model_path': "models/checkers_net.pth",
        'data_path': "data/training_data.pkl",
        'log_dir': "logs",
        'batch_size': 32,
        'learning_rate': 0.001,
        'iterations': 3,              # 生成-训练迭代次数
        'games_per_iteration': 10,    # 每轮生成游戏数
        'epochs_per_iteration': 15    # 每轮训练epoch数
    }

    # 创建训练器
    trainer = CheckersTrainer(
        model_path=config['model_path'],
        data_path=config['data_path'],
        log_dir=config['log_dir'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    try:
        # 运行生成-训练迭代
        trainer.generate_data_and_train(
            iterations=config['iterations'],
            games_per_iteration=config['games_per_iteration'],
            epochs_per_iteration=config['epochs_per_iteration']
        )

        # 最终测试
        trainer.test_model_performance()

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 保存最终模型
        trainer._save_model()

        print("\n" + "=" * 70)
        print(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


if __name__ == "__main__":
    main()