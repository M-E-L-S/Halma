"""
中国跳棋神经网络模型
包含棋盘编码器、神经网络和混合评估器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from typing import Dict, List, Set

# 导入棋盘相关类
try:
    from src.core.board import CubeCoord, ChineseCheckersBoard
    from src.core.moves import ChineseCheckersMoves
except ImportError:
    print("警告: 无法导入棋盘模块，使用简化版本")

    class CubeCoord:
        def __init__(self, q, r, s=None):
            self.q = q
            self.r = r
            self.s = -q - r if s is None else s

        def __eq__(self, other):
            return self.q == other.q and self.r == other.r

        def __hash__(self):
            return hash((self.q, self.r, self.s))

        def distance(self, other):
            return max(abs(self.q - other.q), abs(self.r - other.r), abs(self.s - other.s))

        def neighbor(self, direction):
            # 简化版本
            dirs = [(1, 0, -1), (1, -1, 0), (0, -1, 1),
                   (-1, 0, 1), (-1, 1, 0), (0, 1, -1)]
            q, r, s = dirs[direction % 6]
            return CubeCoord(self.q + q, self.r + r, self.s + s)

class BoardEncoder:
    """棋盘状态编码器 - 将棋盘转换为神经网络输入"""

    def __init__(self, board_size=181):
        self.board_size = board_size

        # 创建参考棋盘用于获取区域信息
        self.reference_board = ChineseCheckersBoard()
        self.all_cells = list(self.reference_board.get_all_cells())

        # 确保我们有足够的格子
        if len(self.all_cells) > board_size:
            print(f"警告: 实际棋盘格子数({len(self.all_cells)})大于设定的board_size({board_size})")
            self.all_cells = self.all_cells[:board_size]
        elif len(self.all_cells) < board_size:
            # 添加虚拟格子
            while len(self.all_cells) < board_size:
                self.all_cells.append(CubeCoord(999, 999))  # 虚拟坐标

        # 预先计算每个格子的特征
        self._precompute_features()

    def _precompute_features(self):
        """预先计算每个格子的静态特征"""
        self.region_features = {}
        self.distance_features = {}

        for coord in self.all_cells:
            if not self.reference_board.is_valid_cell(coord):
                continue

            # 区域特征
            region = self.reference_board.get_region(coord)
            region_code = self._encode_region(region)
            self.region_features[coord] = region_code

            # 距离特征（到各目标区域的距离）
            dist_features = self._compute_distance_features(coord)
            self.distance_features[coord] = dist_features

    def _encode_region(self, region):
        """将区域编码为数值"""
        if region is None:
            return 0.0
        elif region == 'hex':
            return 0.1
        elif region.startswith('tri'):
            try:
                tri_num = int(region[3:])
                return 0.2 + tri_num * 0.1
            except:
                return 0.2
        else:
            return 0.0

    def _compute_distance_features(self, coord):
        """计算坐标到各个目标区域的距离特征"""
        features = np.zeros(6, dtype=np.float32)  # 6个三角形区域

        for player in [1, -1]:
            target_region = self.reference_board.player_target_regions[player]
            target_cells = [c for c, r in self.reference_board.regions.items()
                           if r == target_region]

            if target_cells:
                # 计算到目标区域的平均距离
                distances = [coord.distance(t) for t in target_cells]
                avg_dist = sum(distances) / len(distances)

                # 归一化距离
                normalized_dist = min(avg_dist / 15.0, 1.0)

                # 玩家1和玩家2使用不同的特征通道
                if player == 1:
                    features[0] = 1.0 - normalized_dist
                else:
                    features[1] = 1.0 - normalized_dist

        return features

    def encode_board(self, board_state: Dict[CubeCoord, int],
                    current_player: int) -> np.ndarray:
        """
        编码棋盘状态为神经网络输入

        Args:
            board_state: 棋盘状态字典 {坐标: 玩家}
            current_player: 当前玩家 (1 或 -1)

        Returns:
            特征矩阵 [11, board_size]
        """
        # 11个特征通道
        features = np.zeros((11, self.board_size), dtype=np.float32)

        for idx, coord in enumerate(self.all_cells):
            if idx >= self.board_size:
                break

            # 如果坐标无效，跳过
            if not self.reference_board.is_valid_cell(coord):
                features[0, idx] = -1.0  # 标记无效格子
                continue

            # 通道0: 格子有效性 (1.0表示有效)
            features[0, idx] = 1.0

            # 通道1-2: 棋子位置
            piece = board_state.get(coord, 0)
            if piece == current_player:
                features[1, idx] = 1.0  # 当前玩家棋子
            elif piece != 0:
                features[2, idx] = 1.0  # 对手棋子

            # 通道3-8: 区域特征和距离特征
            if coord in self.region_features:
                features[3, idx] = self.region_features[coord]

            if coord in self.distance_features:
                dist_features = self.distance_features[coord]
                for j in range(6):
                    features[4 + j, idx] = dist_features[j]

            # 通道9: 是否在中心区域
            region = self.reference_board.get_region(coord)
            if region == 'hex':
                features[9, idx] = 1.0

            # 通道10: 是否在边界（简化）
            # 这里可以根据需要添加更复杂的特征

        return features

    def get_board_size(self):
        """获取棋盘大小"""
        return self.board_size


class CheckersNet(nn.Module):
    """中国跳棋神经网络模型"""

    def __init__(self, input_channels=11, board_size=181, hidden_size=256):
        super(CheckersNet, self).__init__()

        # 卷积层提取局部特征
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # 自适应池化提取全局特征
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

        # 全连接层
        pool_output_size = 256 * 8
        self.fc1 = nn.Linear(pool_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, 1)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [batch_size, channels, board_size]

        Returns:
            评估值 [batch_size, 1]
        """
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 全局池化
        x = self.adaptive_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x


class NeuralEvaluator:
    """神经网络评估器 - 替换传统规则评估"""

    def __init__(self, model_path=None, use_gpu=True):
        self.encoder = BoardEncoder()
        self.model = CheckersNet(
            input_channels=11,
            board_size=self.encoder.get_board_size()
        )

        # 设备设置
        self.device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
        self.model.to(self.device)

        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            try:
                if self.device.type == 'cuda':
                    self.model.load_state_dict(torch.load(model_path))
                else:
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"加载神经网络模型: {model_path}")
                self.model.eval()
            except Exception as e:
                print(f"加载模型失败: {e}")

        print(f"神经网络评估器初始化完成，使用设备: {self.device}")

    def evaluate(self, board_state: Dict[CubeCoord, int],
                current_player: int) -> float:
        """
        评估棋盘状态

        Args:
            board_state: 棋盘状态
            current_player: 当前玩家 (1 或 -1)

        Returns:
            评估值，正数对当前玩家有利
        """
        try:
            # 编码棋盘
            features = self.encoder.encode_board(board_state, current_player)

            # 转换为张量
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # 添加批次维度

            # 移动到设备
            features_tensor = features_tensor.to(self.device)

            # 推理
            self.model.eval()
            with torch.no_grad():
                output = self.model(features_tensor)
                score = output.item()

            return score

        except Exception as e:
            print(f"神经网络评估失败: {e}")
            # 返回简单的备用评估
            return self._simple_backup_evaluate(board_state, current_player)

    def _simple_backup_evaluate(self, board_state, current_player):
        """简单的备用评估函数"""
        score = 0.0

        # 棋子计数
        player_pieces = 0
        opponent_pieces = 0

        for piece in board_state.values():
            if piece == current_player:
                player_pieces += 1
            elif piece != 0:
                opponent_pieces += 1

        score += (player_pieces - opponent_pieces) * 10

        return score

    def train_mode(self):
        """切换到训练模式"""
        self.model.train()

    def eval_mode(self):
        """切换到评估模式"""
        self.model.eval()


class HybridEvaluator:
    """混合评估器 - 结合神经网络和规则评估"""

    def __init__(self, model_path=None, neural_weight=0.7):
        self.neural_evaluator = NeuralEvaluator(model_path)
        self.neural_weight = neural_weight
        self.rule_weight = 1.0 - neural_weight

        # 导入规则评估器
        try:
            from evaluator import ChineseCheckersEvaluator
            self.rule_evaluator = ChineseCheckersEvaluator
        except ImportError:
            print("警告: 无法导入规则评估器")
            self.rule_evaluator = None

    def evaluate(self, board_state: Dict[CubeCoord, int],
                current_player: int) -> float:
        """混合评估"""
        # 神经网络评估
        neural_score = self.neural_evaluator.evaluate(board_state, current_player)

        # 规则评估
        if self.rule_evaluator:
            try:
                board = ChineseCheckersBoard()
                rule_eval = self.rule_evaluator(board)
                rule_score = rule_eval.evaluate(board_state, current_player)
            except:
                rule_score = 0.0
        else:
            rule_score = 0.0

        # 归一化分数
        neural_norm = self._sigmoid_normalize(neural_score)
        rule_norm = self._sigmoid_normalize(rule_score / 100.0)  # 规则评估通常范围较大

        # 混合评估
        hybrid_score = (neural_norm * self.neural_weight +
                       rule_norm * self.rule_weight)

        # 转换为传统评估范围（大约-100到100）
        final_score = (hybrid_score - 0.5) * 200

        return final_score

    def _sigmoid_normalize(self, x):
        """Sigmoid归一化"""
        return 1.0 / (1.0 + math.exp(-x))


# 测试函数
def test_model():
    """测试模型"""
    print("测试神经网络模型...")

    # 创建模型
    model = CheckersNet()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试输入
    batch_size = 4
    input_channels = 11
    board_size = 181

    test_input = torch.randn(batch_size, input_channels, board_size)
    output = model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

    return model


if __name__ == "__main__":
    model = test_model()