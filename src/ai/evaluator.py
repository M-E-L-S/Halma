"""
中国跳棋评估函数模块
实现了基于规则和启发式的双人中国跳棋局面评估
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
        self.DISTANCE_WEIGHT = 1.0

        self.MAX_DISTANCE_WEIGHT = 0.5  # 最远棋子距离权重
        self.DISTANCE_LINE_WEIGHT = 1.0  # 平均距离权重
        self.DISTANCE_SQUARED_WEIGHT = 0.5 # 距离平方和权重


        # 棋子分布权重
        self.FORMATE_WEIGHT = 1.0

        self.ISOLATED_PIECE_PENALTY = -2  # 孤立棋子惩罚


        # 连通性权重
        self.CONNECTIVITY_WEIGHT = 1.5

        self.reach_weight = 0.1
        self.depth_weight = 1


        # 进度权重
        self.PROGRESS_WEIGHT = 1.5

        self.LEFT_START_WEIGHT = 5.0  # 离开起始区域的奖励
        self.COMPLETION_WEIGHT = 5.0  # 完成目标的奖励


class ChineseCheckersEvaluator:
    def __init__(self, _board, weights: Optional[EvaluationWeights] = None):
        """
        初始化评估器
        """
        self.board = _board
        self.board_state = None
        self.weights = weights or EvaluationWeights()

        self._initialize_from_board()

    def _initialize_from_board(self):
        """从棋盘对象初始化区域信息"""
        # 获取所有单元格
        self.all_cells = self.board.get_all_cells()

        # 定义目标区域（根据棋盘设置）
        self.player1_target_cells = self._get_target_cells(Player.PLAYER1.value)
        self.player2_target_cells = self._get_target_cells(Player.PLAYER2.value)

        # 定义起始区域
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

        # 提取棋子位置
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

        # 1. 距离评估（到目标区域的距离）
        distance_score = self._evaluate_distance(
            my_positions, my_target, opp_positions, opp_target)
        evaluation += distance_score

        # 2. 阵型评估
        formation_score = self._evaluate_formation(
            my_positions, opp_positions)
        evaluation += formation_score

        # 3. 连通性评估
        connectivity_score = self._evaluate_connectivity(
            my_positions, opp_positions)
        evaluation += connectivity_score

        # 4. 进度评估
        progress_score = self._evaluate_progress(
            my_positions, my_target, opp_positions, opp_target)
        evaluation += progress_score
        print(f"Total Evaluation: {evaluation:.2f}")
        print(f"current_player: {current_player}")

        return evaluation

    def _evaluate_distance(self, my_positions: List[CubeCoord], my_target: Set[CubeCoord],
                          opp_positions: List[CubeCoord], opp_target: Set[CubeCoord]) -> float:
        """
        评估棋子到目标区域的距离
        """
        d_score = 0.0

        # 计算我方棋子到目标区域的平均距离
        my_distances = []
        for pos in my_positions:
            if pos in my_target:
                my_distances.append(0)
            else:
                # 计算到目标区域所有位置的最小距离
                min_dist = float('inf')
                for target in my_target:
                    dist = pos.distance(target)
                    if dist < min_dist:
                        min_dist = dist
                my_distances.append(min_dist)

        # 计算对方棋子到其目标区域的平均距离
        opp_distances = []
        for pos in opp_positions:
            if pos in opp_target:
                opp_distances.append(0)
            else:
                min_dist = float('inf')
                for target in opp_target:
                    dist = pos.distance(target)
                    if dist < min_dist:
                        min_dist = dist
                opp_distances.append(min_dist)

        # 距离评估：对方平均距离 - 我方平均距离
        avg_my_dist = sum(my_distances) / len(my_distances) if my_distances else 0
        avg_opp_dist = sum(opp_distances) / len(opp_distances) if opp_distances else 0

        d_score += (avg_opp_dist - avg_my_dist) * self.weights.DISTANCE_LINE_WEIGHT

        # 额外奖励：最远棋子的距离改善
        if my_distances:
            max_my_dist = max(my_distances)
            d_score -= max_my_dist * self.weights.MAX_DISTANCE_WEIGHT

        # 距离平方和：鼓励均衡前进
        sum_sq_my = sum(d*d for d in my_distances)
        sum_sq_opp = sum(d*d for d in opp_distances)
        d_score += (sum_sq_opp - sum_sq_my) * self.weights.DISTANCE_SQUARED_WEIGHT

        return d_score * self.weights.DISTANCE_WEIGHT

    def _evaluate_formation(self, my_positions: List[CubeCoord],
                           opp_positions: List[CubeCoord]) -> float:
        """
        评估棋子孤立程度
        """
        if len(my_positions) < 2:
            return 0.0

        ave_dists: List[float] = []
        for pos in my_positions:
            pos_dists = [pos.distance(other) for other in my_positions if other != pos]
            ave_dist_score = (sum(pos_dists) / len(pos_dists)) * self.weights.ISOLATED_PIECE_PENALTY
            ave_dists.append(ave_dist_score)

        return (sum(ave_dists) / len(ave_dists)) * self.weights.FORMATE_WEIGHT

    def jump_reach_and_depth(self, start: CubeCoord):
        visited: Set[CubeCoord] = set()
        max_depth = 0
        stack = [(start, 0)]
        while stack:
            cur, _depth = stack.pop()
            for d in range(6):
                mid = cur.neighbor(d)
                landing = mid.neighbor(d)
                # 必须存在格子：中间格被占、落点存在且为空
                if mid in self.board_state and self.board_state[mid] != 0 and \
                        landing in self.board_state and self.board_state[landing] == 0 and \
                        landing not in visited and landing != start:
                    visited.add(landing)
                    stack.append((landing, _depth + 1))
                    if _depth + 1 > max_depth:
                        max_depth = _depth + 1
        return len(visited), max_depth

    def _evaluate_connectivity(self, my_positions: List[CubeCoord],
                               opp_positions: List[CubeCoord]) -> float:
        """
        详细评估棋子可跳跃能力
        我方能跳正分数，敌方能跳负分数
        跳跃距离越远分数越高
        """
        my_reach_total = 0
        my_max_depth_total = 0
        for pos in my_positions:
            reach_count, depth = self.jump_reach_and_depth(pos)
            my_reach_total += reach_count
            my_max_depth_total = max(my_max_depth_total, depth)

        opp_reach_total = 0
        opp_max_depth_total = 0
        for pos in opp_positions:
            reach_count, depth = self.jump_reach_and_depth(pos)
            opp_reach_total += reach_count
            opp_max_depth_total = max(opp_max_depth_total, depth)

        c_score = ((my_reach_total - opp_reach_total) * self.weights.reach_weight + (my_max_depth_total - opp_max_depth_total) * self.weights.depth_weight)

        return c_score * self.weights.CONNECTIVITY_WEIGHT

    def _evaluate_progress(self, my_positions: List[CubeCoord], my_target: Set[CubeCoord],
                           opp_positions: List[CubeCoord], opp_target: Set[CubeCoord]) -> float:
        """
        评估游戏进度
        已完成目标、接近目标的棋子等
        """
        p_score = 0.0

        # 计算已到达目标区域的棋子数量
        my_in_target = sum(1 for pos in my_positions if pos in my_target)
        opp_in_target = sum(1 for pos in opp_positions if pos in opp_target)

        # 完成目标奖励
        p_score += (my_in_target - opp_in_target) * self.weights.COMPLETION_WEIGHT

        # 奖励离开起始区域的棋子
        my_left_start = sum(1 for pos in my_positions if pos not in self.player1_start_cells)
        p_score += my_left_start * self.weights.LEFT_START_WEIGHT

        return p_score * self.weights.PROGRESS_WEIGHT

    def quick_evaluate(self, current_player: Player) -> float:
        """
        快速评估函数

        用于搜索树的浅层评估，计算速度更快
        只考虑关键因素：距离、进度和位置价值
        """
        # 提取棋子位置
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