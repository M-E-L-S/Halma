"""
中国跳棋评估函数模块
实现了基于规则和启发式的双人中国跳棋局面评估

author: 徐温洌
"""

from typing import Dict, List, Optional, Set
from enum import Enum
from src.core.board import CubeCoord, ChineseCheckersBoard

class Player(Enum):
    """玩家枚举"""
    PLAYER1 = 1
    PLAYER2 = -1

class EvaluationWeights:
    """评估权重配置"""
    def __init__(self):
        # 基础距离权重
        self.DISTANCE_WEIGHT = 10

        self.DISTANCE_SQUARED_WEIGHT = 1 # 距离平方和权重，量级10


        # 棋子分布权重
        self.FORMATE_WEIGHT = 1

        self.ISOLATED_PIECE_PENALTY = -1  # 孤立棋子惩罚，量级10


        # 连通性权重
        self.CONNECTIVITY_WEIGHT = 0.5

        self.depth_weight = 10  # 向前跳跃深度权重，量级10


        # 进度权重
        self.PROGRESS_WEIGHT = 0.1

        self.LEFT_START_WEIGHT = 10.0  # 离开起始区域的奖励，量级10
        self.COMPLETION_WEIGHT = 10.0  # 完成目标的奖励，量级10


class ChineseCheckersEvaluator:
    def __init__(self, _board, weights: Optional[EvaluationWeights] = None):
        """初始化评估器"""
        self.board = _board
        self.board_state = None
        self.weights = weights or EvaluationWeights()

        self._initialize_from_board()

    def _initialize_from_board(self):
        """从棋盘对象初始化区域信息"""
        self.all_cells = self.board.get_all_cells()

        self.player1_target_cells = self._get_target_cells(Player.PLAYER1.value)
        self.player2_target_cells = self._get_target_cells(Player.PLAYER2.value)
        self.player1_start_cells = self._get_start_cells(Player.PLAYER1.value)
        self.player2_start_cells = self._get_start_cells(Player.PLAYER2.value)

    def _get_target_cells(self, player: int) -> Set[CubeCoord]:
        """获取玩家的目标区域单元格"""
        target_region = self.board.player_target_regions[player]
        return {coord for coord in self.all_cells
                if self.board.regions.get(coord) == target_region}

    def _get_start_cells(self, player: int) -> Set[CubeCoord]:
        """获取玩家的起始区域单元格"""
        start_region = self.board.player_start_regions[player]
        return {coord for coord in self.all_cells
                if self.board.regions.get(coord) == start_region}

    def evaluate(self, _board_state: Dict[CubeCoord, int], current_player: int) -> float:
        """
        综合评估函数

        Args:
            _board_state: 棋盘状态字典，格式为 {CubeCoord: player_id}
            current_player: 当前玩家

        Returns:
            评估值，正数对当前玩家有利
        """
        self.board_state = _board_state

        player1_positions = [pos for pos, player in self.board_state.items()
                           if player == Player.PLAYER1.value]
        player2_positions = [pos for pos, player in self.board_state.items()
                           if player == Player.PLAYER2.value]

        if current_player == Player.PLAYER1.value:
            my_positions = player1_positions
            opp_positions = player2_positions
            my_target = self.player1_target_cells
            opp_target = self.player2_target_cells
        else:
            my_positions = player2_positions
            opp_positions = player1_positions
            my_target = self.player2_target_cells
            opp_target = self.player1_target_cells

        # 计算各项评估指标
        evaluation = 0.0

        # 1. 距离评估
        distance_score = self._evaluate_distance(
            my_positions, my_target, opp_positions, opp_target)
        evaluation += distance_score

        # 2. 阵型评估
        formation_score = self._evaluate_formation(
            my_positions, opp_positions, my_target)
        evaluation += formation_score

        # 3. 连通性评估
        connectivity_score = self._evaluate_connectivity(
            my_positions, opp_positions, my_target, opp_target)
        evaluation += connectivity_score

        # 4. 进度评估
        progress_score = self._evaluate_progress(
            my_positions, my_target, opp_target)
        evaluation += progress_score

        return evaluation

    def _evaluate_distance(self, my_positions: List[CubeCoord], my_target: Set[CubeCoord],
                          opp_positions: List[CubeCoord], opp_target: Set[CubeCoord]) -> float:
        """
        评估棋子到目标区域的距离
        """
        d_score = 0.0

        # 计算双方棋子到目标区域的距离列表
        # 不使用线性，使用距离平方和，保证均衡前进。
        my_distances = []
        for pos in my_positions:
            if pos in my_target:
                my_distances.append(0)
            else:
                d = pos.distance(next(iter(my_target)))
                my_distances.append(d*d)

        opp_distances = []
        for pos in opp_positions:
            if pos in opp_target:
                opp_distances.append(0)
            else:
                d = pos.distance(next(iter(opp_target)))
                opp_distances.append(d*d)

        avg_my = sum(my_distances) / len(my_distances) if my_distances else 0
        avg_opp = sum(opp_distances) / len(opp_distances) if opp_distances else 0
        d_score += (0.9 * avg_opp - avg_my) * self.weights.DISTANCE_SQUARED_WEIGHT

        return d_score * self.weights.DISTANCE_WEIGHT

    def _evaluate_formation(self, my_positions: List[CubeCoord], opp_positions: List[CubeCoord], my_target: Set[CubeCoord]) -> float:
        """
        孤立棋子惩罚
        """
        i_score = 0.0
        avg_dist = 0.0

        for pos in my_positions:
            if pos not in my_target:
                d = pos.distance(next(iter(my_target)))
                avg_dist += d
        avg_dist = avg_dist / len(my_positions) if my_positions else 0

        #给落在己方棋子之后的孤立棋子惩罚，深入敌阵的不惩罚
        for pos in my_positions:
            if pos not in my_target:
                positions = my_positions + opp_positions
                min_dists = min([pos.distance(other) for other in positions])
                if min_dists > 1 and pos.distance(next(iter(my_target))) > avg_dist:
                    i_score += min_dists * 10 * self.weights.ISOLATED_PIECE_PENALTY

        return i_score * self.weights.FORMATE_WEIGHT

    def jump_reach_and_depth(self, start: CubeCoord, my_target: Set[CubeCoord]):
        visited: Set[CubeCoord] = set()
        max_depth = 0
        stack = [(start, 0)]
        dist = start.distance(next(iter(my_target)))
        while stack:
            cur, _depth = stack.pop()
            for d in range(6):
                mid = cur.neighbor(d)
                landing = mid.neighbor(d)
                if mid in self.board_state and self.board_state[mid] != 0 and \
                        landing in self.board_state and self.board_state[landing] == 0 and \
                        landing not in visited and landing != start:
                    visited.add(landing)
                    stack.append((landing, _depth + 1))
                    if _depth + 1 > max_depth and landing.distance(next(iter(my_target))) < dist:
                        max_depth = _depth + 1
        return max_depth

    def _evaluate_connectivity(self, my_positions: List[CubeCoord],
                               opp_positions: List[CubeCoord], my_target: Set[CubeCoord],
                               opp_target: Set[CubeCoord]) -> float:
        """
        详细评估棋子可跳跃能力
        我方能跳正分数，敌方能跳负分数
        跳跃距离越远分数越高
        """
        my_positions = [pos for pos in my_positions if pos not in my_target]
        opp_positions = [pos for pos in opp_positions if pos not in opp_target]

        my_max_depth_total = 0
        for pos in my_positions:
            my_max_depth_total += self.jump_reach_and_depth(pos, my_target)

        opp_max_depth_total = 0
        for pos in opp_positions:
            opp_max_depth_total += self.jump_reach_and_depth(pos, opp_target)

        c_score = (my_max_depth_total - 0.9 * opp_max_depth_total) * self.weights.depth_weight

        return c_score * self.weights.CONNECTIVITY_WEIGHT

    def _evaluate_progress(self, my_positions: List[CubeCoord], my_target: Set[CubeCoord], opp_target: Set[CubeCoord]) -> float:
        """
        评估游戏进度
        已完成目标、接近目标的棋子等
        """
        p_score = 0.0

        # 计算已到达目标区域的棋子数量
        my_in_target = sum(1 for pos in my_positions if pos in my_target)

        # 完成目标奖励
        p_score += my_in_target * self.weights.COMPLETION_WEIGHT

        # 奖励离开起始区域的棋子
        my_left_start = sum(1 for pos in my_positions if pos not in opp_target)
        p_score += my_left_start * self.weights.LEFT_START_WEIGHT

        return p_score * self.weights.PROGRESS_WEIGHT

    def quick_evaluate(self, current_player: Player) -> float:
        """
        快速评估函数

        用于搜索树的浅层评估，计算速度更快
        只考虑关键因素：距离、进度和位置价值
        """
        player1_positions = [pos for pos, player in board_state.items()
                           if player == Player.PLAYER1.value]
        player2_positions = [pos for pos, player in board_state.items()
                           if player == Player.PLAYER2.value]

        if current_player == Player.PLAYER1:
            my_positions = player1_positions
            opp_positions = player2_positions
            my_target = self.player1_target_cells
            opp_target = self.player2_target_cells
        else:
            my_positions = player2_positions
            opp_positions = player1_positions
            my_target = self.player2_target_cells
            opp_target = self.player1_target_cells

        # 快速评估只考虑关键因素
        score = 0.0

        # 1. 距离评估
        my_avg_dist = self._average_distance_to_target(my_positions, my_target)
        opp_avg_dist = self._average_distance_to_target(opp_positions, opp_target)
        score += (opp_avg_dist - my_avg_dist) * 1.5

        # 2. 进度评估
        my_in_target = sum(1 for pos in my_positions if pos in my_target)
        opp_in_target = sum(1 for pos in opp_positions if pos in opp_target)
        score += (my_in_target - opp_in_target) * 2.0

        return score

    def _average_distance_to_target(self, positions: List[CubeCoord], target: Set[CubeCoord]) -> float:
        """计算到目标区域的平均距离"""
        if not positions:
            return 0.0

        total_dist = 0
        for pos in positions:
            if pos in target:
                continue
            min_dist = min(pos.distance(t) for t in target)
            total_dist += min_dist

        return total_dist / len(positions)


# 使用示例
if __name__ == "__main__":
    # 创建棋盘
    board = ChineseCheckersBoard()

    # 创建评估器
    evaluator = ChineseCheckersEvaluator(board)

    # 使用当前棋盘状态
    board_state = board.cells.copy()  # 获取当前棋盘状态

    # 评估当前局面（从玩家1的角度）
    score = evaluator.evaluate(board_state, 1)
    print(f"玩家1的评估分数: {score:.2f}")

    # 快速评估
    quick_score = evaluator.quick_evaluate(board_state)
    print(f"玩家1的快速评估分数: {quick_score:.2f}")