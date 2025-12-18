# python
"""
真实自对弈数据生成器（使用 AI 进行自对弈并生成训练数据）
基于真实的中国跳棋规则生成训练数据
"""

import torch
import numpy as np
import random
import pickle
import os
import math
import time
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional

# 导入必要的模块
try:
    from src.core.board import CubeCoord, ChineseCheckersBoard
    from src.core.moves import ChineseCheckersMoves
    from src.ai.tradition_serach import Search
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确目录下运行")

# 数据样本定义
TrainingSample = namedtuple('TrainingSample',
                          ['features', 'value_target', 'policy_target', 'player'])

class RealSelfPlayGenerator:
    """真实自对弈数据生成器（优先使用 AI）"""

    def __init__(self,
                 model_path: str = None,
                 games_per_generation: int = 50,
                 max_turns: int = 200,
                 use_ai: bool = True,
                 ai_depth: int = 2):
        """
        初始化自对弈生成器
        """
        self.games_per_generation = games_per_generation
        self.max_turns = max_turns
        self.use_ai = use_ai
        self.ai_depth = ai_depth

        # 编码器
        from model import BoardEncoder
        self.encoder = BoardEncoder()

        # 神经网络评估器（可选）
        if model_path and os.path.exists(model_path):
            from model import NeuralEvaluator
            self.neural_evaluator = NeuralEvaluator(model_path)
        else:
            self.neural_evaluator = None

        # 数据存储
        self.training_data = []
        self.game_history = []

        # 统计信息
        self.stats = {
            'total_games': 0,
            'total_moves': 0,
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'avg_turns': 0,
            'total_samples': 0
        }

        # AI代理
        self.ai_agents = {}
        if self.use_ai:
            try:
                self.ai_agents[1] = Search(player=1, depth=ai_depth)
                self.ai_agents[-1] = Search(player=-1, depth=ai_depth)
            except Exception as e:
                print(f"警告: AI初始化失败 ({e})，将使用随机策略")
                self.use_ai = False

    def _get_ai_move(self, board: ChineseCheckersBoard, player: int) -> Tuple[Optional[List[CubeCoord]], Optional[np.ndarray]]:
        """
        获取AI移动及其策略向量（若AI能提供）
        返回 (move, policy_vector)：
          - move: 选定的一步（List[CubeCoord]）或 None
          - policy_vector: 与 encoder.get_board_size() 等长的概率向量（或 None）
        """
        if not self.use_ai or player not in self.ai_agents:
            return None, None

        try:
            # 创建游戏状态
            game_state = {
                'board': board,
                'current_player': player,
                'valid_moves': ChineseCheckersMoves.generate_all_moves(board, player)
            }

            ai_agent = self.ai_agents[player]

            # 有些 Search 实现可能返回 (move, policy) 或者单纯 return move
            result = ai_agent.make_move(game_state)

            move = None
            policy_vec = None

            # 如果返回一个二元组，尝试解析
            if isinstance(result, tuple) and len(result) == 2:
                move, policy_vec = result
                # 确保 policy_vec 是 numpy 数组且长度与 encoder 一致
                if policy_vec is not None:
                    policy_vec = np.asarray(policy_vec, dtype=float)
                    # 若长度不匹配，尝试调整或丢弃
                    if policy_vec.shape[0] != self.encoder.get_board_size():
                        policy_vec = None
            else:
                # 仅返回 move
                move = result

            # 若没有得到 policy_vec，构造基于合法走法的概率向量（one-hot 或均匀分布）
            if move is not None and policy_vec is None:
                legal_moves = game_state['valid_moves']
                policy_vec = np.zeros(self.encoder.get_board_size(), dtype=float)

                try:
                    idx = legal_moves.index(move)
                    # 将 idx 映射到 policy 向量的前几位（若 encoder 空间足够）
                    if idx < policy_vec.shape[0]:
                        policy_vec[idx] = 1.0
                    else:
                        # 若索引超出范围，退为均匀分布在合法走法数量上的前 N 项
                        policy_vec[:min(len(legal_moves), policy_vec.shape[0])] = 1.0 / min(len(legal_moves), policy_vec.shape[0])
                except Exception:
                    # 无法找到 move 在 legal_moves 中，退为均匀分布
                    n = min(len(legal_moves), policy_vec.shape[0]) if len(legal_moves) > 0 else 1
                    if n > 0:
                        policy_vec[:n] = 1.0 / n

            return move, policy_vec

        except Exception as e:
            print(f"AI移动失败: {e}")
            return None, None

    def _get_random_move(self, board: ChineseCheckersBoard, player: int) -> Tuple[Optional[List[CubeCoord]], Optional[np.ndarray]]:
        """获取随机移动并返回简单策略向量"""
        moves = ChineseCheckersMoves.generate_all_moves(board, player)

        if not moves:
            return None, None

        move = random.choice(moves)
        policy = np.zeros(self.encoder.get_board_size(), dtype=float)
        try:
            idx = moves.index(move)
            if idx < policy.shape[0]:
                policy[idx] = 1.0
            else:
                policy[:1] = 1.0
        except Exception:
            policy[:1] = 1.0

        return move, policy

    def _evaluate_position(self, board: ChineseCheckersBoard, player: int) -> float:
        """评估位置（用于生成训练目标）"""
        if self.neural_evaluator:
            # 使用神经网络评估
            try:
                return self.neural_evaluator.evaluate(board.cells.copy(), player)
            except Exception:
                pass

        # 备用：简单规则评估
        score = 0.0

        # 棋子计数
        player_pieces = len(board.get_player_pieces(player))
        opponent_pieces = len(board.get_player_pieces(-player))

        score += (player_pieces - opponent_pieces) * 10

        # 进度评估
        target_region = board.player_target_regions[player]
        player_pieces_list = board.get_player_pieces(player)

        pieces_in_target = 0
        for piece in player_pieces_list:
            if board.get_region(piece) == target_region:
                pieces_in_target += 1

        score += pieces_in_target * 20

        return score

    def _check_game_over(self, board: ChineseCheckersBoard) -> Tuple[bool, Optional[int]]:
        """检查游戏是否结束，返回(是否结束, 获胜者)"""
        # 检查玩家1是否获胜
        player1_won = True
        target_region = board.player_target_regions[1]
        player1_pieces = board.get_player_pieces(1)

        if len(player1_pieces) != 10:
            player1_won = False
        else:
            for coord in player1_pieces:
                if board.get_region(coord) != target_region:
                    player1_won = False
                    break

        if player1_won:
            return True, 1

        # 检查玩家2是否获胜
        player2_won = True
        target_region = board.player_target_regions[-1]
        player2_pieces = board.get_player_pieces(-1)

        if len(player2_pieces) != 10:
            player2_won = False
        else:
            for coord in player2_pieces:
                if board.get_region(coord) != target_region:
                    player2_won = False
                    break

        if player2_won:
            return True, -1

        return False, None

    def generate_game(self, game_idx: int = 0) -> List[TrainingSample]:
        """生成一局真实的自对弈数据（AI vs AI 或 AI vs 随机）"""
        print(f"生成游戏 {game_idx + 1}/{self.games_per_generation}")

        # 初始化棋盘
        board = ChineseCheckersBoard()
        current_player = 1
        game_samples = []
        turn_count = 0

        # 游戏循环
        while turn_count < self.max_turns:
            # 检查游戏是否结束
            game_over, winner = self._check_game_over(board)
            if game_over:
                break

            # 获取所有合法移动
            moves = ChineseCheckersMoves.generate_all_moves(board, current_player)

            if not moves:
                # 没有合法移动，切换玩家
                current_player *= -1
                continue

            # 选择移动策略：优先使用AI
            if self.use_ai:
                move, policy = self._get_ai_move(board, current_player)
                if not move:
                    move, policy = self._get_random_move(board, current_player)
            else:
                move, policy = self._get_random_move(board, current_player)

            if not move:
                # 没有找到移动，跳过
                current_player *= -1
                continue

            # 编码当前局面并创建样本
            try:
                features = self.encoder.encode_board(board.cells.copy(), current_player)

                # 评估当前局面（用于初始 value target，后续按胜负修正）
                position_value = self._eigmoid_normalize(
                    self._evaluate_position(board, current_player)
                )

                # 如果策略向量为 None，则创建默认均匀策略
                if policy is None:
                    policy = np.zeros(self.encoder.get_board_size(), dtype=float)
                    n = min(len(moves), policy.shape[0]) if len(moves) > 0 else 1
                    if n > 0:
                        policy[:n] = 1.0 / n

                # 创建训练样本
                sample = TrainingSample(
                    features=features.copy(),
                    value_target=position_value,
                    policy_target=policy.copy(),
                    player=current_player
                )

                game_samples.append(sample)

                # 执行移动（注意 apply_move 返回新的 board）
                board = ChineseCheckersMoves.apply_move(board, move)

                # 切换玩家
                current_player *= -1
                turn_count += 1

            except Exception as e:
                print(f"生成样本失败: {e}")
                break

        # 游戏结束，确定结果
        game_over, winner = self._check_game_over(board)

        if game_over:
            # 根据获胜者为所有样本分配最终值
            for i, sample in enumerate(game_samples):
                player = sample.player

                if winner == player:
                    final_value = 1.0  # 获胜
                elif winner == -player:
                    final_value = -1.0  # 失败
                else:
                    final_value = 0.0  # 平局（理论上不会发生）

                # 更新样本的值目标
                game_samples[i] = TrainingSample(
                    features=sample.features,
                    value_target=final_value,
                    policy_target=sample.policy_target,
                    player=sample.player
                )

            # 更新统计
            if winner == 1:
                self.stats['player1_wins'] += 1
            elif winner == -1:
                self.stats['player2_wins'] += 1
            else:
                self.stats['draws'] += 1
        else:
            # 未完成游戏，使用最后评估
            print(f"游戏未完成 (达到最大回合数: {self.max_turns})")
            self.stats['draws'] += 1

        # 更新统计
        self.stats['total_games'] += 1
        self.stats['total_moves'] += turn_count
        self.stats['total_samples'] += len(game_samples)

        avg_turns = self.stats['total_moves'] / max(self.stats['total_games'], 1)
        self.stats['avg_turns'] = avg_turns

        print(f"  游戏结果: {'玩家1获胜' if winner == 1 else '玩家2获胜' if winner == -1 else '平局'}")
        print(f"  回合数: {turn_count}, 生成样本: {len(game_samples)}")

        return game_samples

    def _eigmoid_normalize(self, x: float) -> float:
        """Sigmoid 归一化（保留原设计但命名统一）"""
        return 1.0 / (1.0 + math.exp(-x / 100.0))

    def generate_batch(self, batch_size: int = None) -> List[TrainingSample]:
        """生成一批自对弈数据"""
        if batch_size is None:
            batch_size = self.games_per_generation * 10  # 估计值

        print(f"开始生成 {self.games_per_generation} 局自对弈数据...")
        print(f"目标样本数: {batch_size}")

        start_time = time.time()

        # 清空旧数据（按需求保留或清空）
        self.training_data = []
        self.game_history = []

        games_generated = 0
        samples_generated = 0

        while samples_generated < batch_size and games_generated < self.games_per_generation:
            try:
                # 生成一局游戏
                game_samples = self.generate_game(games_generated)

                if game_samples:
                    # 添加到训练数据
                    self.training_data.extend(game_samples)
                    samples_generated += len(game_samples)
                    games_generated += 1

                    # 保存游戏历史
                    self.game_history.append({
                        'game_idx': games_generated,
                        'turns': len(game_samples),
                        'samples': len(game_samples),
                        'timestamp': time.time()
                    })

                    print(f"  累计样本: {samples_generated}/{batch_size}")

            except Exception as e:
                print(f"生成游戏失败: {e}")
                continue

        elapsed_time = time.time() - start_time

        print(f"\n数据生成完成:")
        print(f"  生成游戏: {games_generated}")
        print(f"  总样本数: {len(self.training_data)}")
        print(f"  耗时: {elapsed_time:.1f}秒")
        print(f"  统计信息: {self.stats}")

        # 返回所有数据
        return self.training_data

    # python
    def save_training_data(self, filepath: str):
        """保存训练数据"""
        # 只在有目录路径时创建目录（避免 os.path.dirname 返回空字符串导致 makedirs 失败）
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # 准备保存的数据
        save_data = {
            'samples': self.training_data,
            'stats': self.stats,
            'game_history': self.game_history,
            'encoder_info': {
                'board_size': self.encoder.get_board_size(),
                'timestamp': time.time()
            }
        }

        # 保存
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"训练数据已保存到: {filepath}")
        print(f"  样本数量: {len(self.training_data)}")
        print(f"  文件大小: {os.path.getsize(filepath) / 1024:.1f} KB")

    def load_training_data(self, filepath: str):
        """加载训练数据"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                self.training_data = data.get('samples', [])
                self.stats = data.get('stats', self.stats)
                self.game_history = data.get('game_history', [])

                print(f"训练数据已从 {filepath} 加载")
                print(f"  样本数量: {len(self.training_data)}")
                print(f"  游戏历史: {len(self.game_history)} 局")

            except Exception as e:
                print(f"加载训练数据失败: {e}")
        else:
            print(f"文件 {filepath} 不存在")


# 测试函数
def test_self_play():
    """测试自对弈生成器（使用 AI）"""
    print("测试自对弈生成器...")

    # 创建生成器（使用 AI）
    generator = RealSelfPlayGenerator(
        games_per_generation=2,
        max_turns=50,
        use_ai=False  # 使用 AI 自对弈
    )

    # 生成少量数据
    batch = generator.generate_batch(batch_size=20)

    if batch:
        sample = batch[0]
        print(f"\n样本信息:")
        print(f"  特征形状: {getattr(sample.features, 'shape', None)}")
        print(f"  值目标: {sample.value_target}")
        print(f"  玩家: {sample.player}")

        # 保存测试数据
        generator.save_training_data("test_self_play_data.pkl")

    return generator


if __name__ == "__main__":
    test_self_play()
