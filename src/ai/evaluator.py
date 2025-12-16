"""
中国跳棋评估函数模块
实现了基于规则和启发式的双人中国跳棋局面评估
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


class Player(Enum):
    """玩家枚举"""
    PLAYER1 = 1
    PLAYER2 = 2


class EvaluationWeights:
    """评估权重配置"""

    def __init__(self):
        # 基础距离权重
        self.DISTANCE_WEIGHT = 1.0
        self.DISTANCE_SQUARED_WEIGHT = 0.3

        # 棋子分布权重
        self.FORMATION_WEIGHT = 0.8
        self.CONNECTIVITY_WEIGHT = 0.5
        self.CLUSTER_WEIGHT = -0.3  # 聚类为负，鼓励分散

        # 位置优势权重
        self.CENTER_CONTROL_WEIGHT = 0.4
        self.BLOCKING_WEIGHT = 0.6
        self.MOBILITY_WEIGHT = 0.7

        # 进度权重
        self.PROGRESS_WEIGHT = 1.2
        self.COMPLETION_WEIGHT = 2.0  # 完成目标的奖励

        # 战略权重
        self.BRIDGE_FORMATION_WEIGHT = 0.4
        self.PATH_CLEARANCE_WEIGHT = 0.5
        self.THREAT_WEIGHT = 0.3


class ChineseCheckersEvaluator:
    """
    中国跳棋评估函数类
    使用多种启发式方法评估双人中国跳棋局面
    """

    def __init__(self, board_size: int = 17, weights: Optional[EvaluationWeights] = None):
        """
        初始化评估器

        Args:
            board_size: 棋盘大小（标准中国跳棋为17）
            weights: 评估权重，如为None则使用默认权重
        """
        self.board_size = board_size
        self.weights = weights or EvaluationWeights()

        # 预计算棋盘距离和位置价值
        self._initialize_board_properties()

        # 目标区域定义（双人版）
        self._define_target_regions()

    def _initialize_board_properties(self):
        """初始化棋盘相关属性"""
        # 棋盘坐标映射
        self.coordinates = {}
        self.inverse_coordinates = {}

        # 六边形方向：6个相邻方向
        self.directions = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]

        # 距离矩阵（预计算）
        self.distance_cache = {}

    def _define_target_regions(self):
        """定义双人目标区域"""
        # 在中国跳棋双人版中，每个玩家的目标区域是对角线对面的三角形区域

        # 玩家1（通常为红色）目标区域：右下角三角形
        self.player1_target = set()
        for i in range(4):  # 三角形大小为4行
            row = self.board_size - 1 - i
            for j in range(i + 1):
                col = self.board_size - 1 - (i - j)
                self.player1_target.add((row, col))

        # 玩家2（通常为蓝色）目标区域：左上角三角形
        self.player2_target = set()
        for i in range(4):
            for j in range(i + 1):
                self.player2_target.add((i, j))

    def evaluate(self, board_state: Dict, current_player: Player) -> float:
        """
        综合评估函数

        Args:
            board_state: 棋盘状态字典，格式为 {position: player_id}
            current_player: 当前玩家

        Returns:
            评估值，正数对当前玩家有利
        """
        # 提取棋子位置
        player1_positions = [pos for pos, player in board_state.items()
                             if player == Player.PLAYER1.value]
        player2_positions = [pos for pos, player in board_state.items()
                             if player == Player.PLAYER2.value]

        if current_player == Player.PLAYER1:
            my_positions = player1_positions
            opp_positions = player2_positions
            my_target = self.player1_target
            opp_target = self.player2_target
        else:
            my_positions = player2_positions
            opp_positions = player1_positions
            my_target = self.player2_target
            opp_target = self.player1_target

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

        # 4. 控制区域评估
        control_score = self._evaluate_control(
            my_positions, opp_positions)
        evaluation += control_score

        # 5. 进度评估
        progress_score = self._evaluate_progress(
            my_positions, my_target, opp_positions, opp_target)
        evaluation += progress_score

        # 6. 阻塞和威胁评估
        threat_score = self._evaluate_threats(
            my_positions, opp_positions, my_target, opp_target)
        evaluation += threat_score

        # 7. 移动性评估
        mobility_score = self._evaluate_mobility(
            my_positions, opp_positions, board_state)
        evaluation += mobility_score

        return evaluation

    def _evaluate_distance(self, my_positions: List[Tuple], my_target: set,
                           opp_positions: List[Tuple], opp_target: set) -> float:
        """
        评估棋子到目标区域的距离

        距离越小越好，使用多种距离度量
        """
        score = 0.0

        # 计算我方棋子到目标区域的平均距离
        my_distances = []
        for pos in my_positions:
            # 如果已经在目标区域，距离为0
            if pos in my_target:
                my_distances.append(0)
            else:
                # 计算到目标区域所有位置的曼哈顿距离，取最小值
                min_dist = float('inf')
                for target in my_target:
                    dist = self._hex_distance(pos, target)
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
                    dist = self._hex_distance(pos, target)
                    if dist < min_dist:
                        min_dist = dist
                opp_distances.append(min_dist)

        # 距离评估：对方平均距离 - 我方平均距离
        avg_my_dist = sum(my_distances) / len(my_distances) if my_distances else 0
        avg_opp_dist = sum(opp_distances) / len(opp_distances) if opp_distances else 0

        score += (avg_opp_dist - avg_my_dist) * self.weights.DISTANCE_WEIGHT

        # 额外奖励：最远棋子的距离改善（鼓励移动落后的棋子）
        if my_distances:
            max_my_dist = max(my_distances)
            score -= max_my_dist * 0.1  # 减少最远距离的惩罚

        # 距离平方和：鼓励均衡前进
        sum_sq_my = sum(d * d for d in my_distances)
        sum_sq_opp = sum(d * d for d in opp_distances)
        score += (sum_sq_opp - sum_sq_my) * self.weights.DISTANCE_SQUARED_WEIGHT

        return score

    def _evaluate_formation(self, my_positions: List[Tuple],
                            opp_positions: List[Tuple]) -> float:
        """
        评估棋子阵型

        好的阵型应该：
        1. 棋子之间保持适当距离（不太近也不太远）
        2. 形成链式结构便于跳跃
        3. 避免过于分散
        """
        score = 0.0

        # 计算我方棋子之间的平均距离
        if len(my_positions) > 1:
            my_pairwise_dists = []
            for i in range(len(my_positions)):
                for j in range(i + 1, len(my_positions)):
                    dist = self._hex_distance(my_positions[i], my_positions[j])
                    my_pairwise_dists.append(dist)

            avg_my_dist = sum(my_pairwise_dists) / len(my_pairwise_dists)

            # 理想距离范围：3-8个六边形单位
            # 距离太近（<3）会阻塞，距离太远（>8）会失去联系
            if avg_my_dist < 3:
                score -= (3 - avg_my_dist) * 0.5
            elif avg_my_dist > 8:
                score -= (avg_my_dist - 8) * 0.3
            else:
                score += 0.5  # 良好阵型奖励

        # 检查是否形成"桥"结构（便于连续跳跃）
        bridge_score = self._check_bridge_formation(my_positions)
        score += bridge_score * self.weights.BRIDGE_FORMATION_WEIGHT

        # 检查棋子聚类程度
        clustering_score = self._evaluate_clustering(my_positions)
        score += clustering_score * self.weights.CLUSTER_WEIGHT

        return score * self.weights.FORMATION_WEIGHT

    def _evaluate_connectivity(self, my_positions: List[Tuple],
                               opp_positions: List[Tuple]) -> float:
        """
        评估棋子连通性

        连通性好的棋子更容易形成连续跳跃
        """
        if not my_positions:
            return 0.0

        # 使用BFS检查连通分量
        visited = set()
        components = []

        for pos in my_positions:
            if pos not in visited:
                # 开始新的连通分量
                component = []
                stack = [pos]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        # 检查所有相邻位置是否有我方棋子
                        for dx, dy in self.directions:
                            neighbor = (current[0] + dx, current[1] + dy)
                            if neighbor in my_positions and neighbor not in visited:
                                stack.append(neighbor)

                components.append(component)

        # 评估连通性：最大连通分量的大小
        if components:
            max_component_size = max(len(comp) for comp in components)
            total_pieces = len(my_positions)

            # 连通性得分：最大连通分量占总棋子数的比例
            connectivity = max_component_size / total_pieces

            # 理想情况是所有棋子都连通
            score = connectivity - 0.5  # 归一化到[-0.5, 0.5]

            # 额外奖励：如果所有棋子都连通
            if len(components) == 1:
                score += 0.3
        else:
            score = 0.0

        return score * self.weights.CONNECTIVITY_WEIGHT

    def _evaluate_control(self, my_positions: List[Tuple],
                          opp_positions: List[Tuple]) -> float:
        """
        评估棋盘控制

        控制中心区域和关键路径
        """
        score = 0.0

        # 计算棋盘中心
        center_row = self.board_size // 2
        center_col = self.board_size // 2

        # 评估中心控制
        center_radius = 3
        center_control = 0

        for pos in my_positions:
            dist_to_center = self._hex_distance(pos, (center_row, center_col))
            if dist_to_center <= center_radius:
                center_control += 1

        # 中心控制得分
        score += (center_control / len(my_positions)) * self.weights.CENTER_CONTROL_WEIGHT

        # 评估关键路径控制
        # 在中国跳棋中，中间列是重要路径
        middle_col = self.board_size // 2
        path_control = 0

        for pos in my_positions:
            if abs(pos[1] - middle_col) <= 1:  # 中间列及其相邻列
                path_control += 1

        score += (path_control / len(my_positions)) * 0.3

        return score

    def _evaluate_progress(self, my_positions: List[Tuple], my_target: set,
                           opp_positions: List[Tuple], opp_target: set) -> float:
        """
        评估游戏进度

        已完成目标、接近目标的棋子等
        """
        score = 0.0

        # 计算已到达目标区域的棋子数量
        my_in_target = sum(1 for pos in my_positions if pos in my_target)
        opp_in_target = sum(1 for pos in opp_positions if pos in opp_target)

        # 完成目标奖励
        score += (my_in_target - opp_in_target) * self.weights.COMPLETION_WEIGHT

        # 计算接近目标区域的棋子（距离<=2）
        my_near_target = 0
        for pos in my_positions:
            if pos not in my_target:
                min_dist = min(self._hex_distance(pos, target) for target in my_target)
                if min_dist <= 2:
                    my_near_target += 1

        opp_near_target = 0
        for pos in opp_positions:
            if pos not in opp_target:
                min_dist = min(self._hex_distance(pos, target) for target in opp_target)
                if min_dist <= 2:
                    opp_near_target += 1

        score += (my_near_target - opp_near_target) * 0.5

        return score * self.weights.PROGRESS_WEIGHT

    def _evaluate_threats(self, my_positions: List[Tuple], opp_positions: List[Tuple],
                          my_target: set, opp_target: set) -> float:
        """
        评估威胁和阻塞

        阻塞对手前进路径的能力
        """
        score = 0.0

        # 检查是否阻塞了对手的关键路径
        blocking_positions = 0

        # 对于每个对手棋子，检查其到目标的最短路径是否被我方棋子阻塞
        for opp_pos in opp_positions:
            if opp_pos in opp_target:
                continue  # 已经在目标区域，不计算阻塞

            # 找到最近的对手目标
            nearest_target = None
            min_dist = float('inf')
            for target in opp_target:
                dist = self._hex_distance(opp_pos, target)
                if dist < min_dist:
                    min_dist = dist
                    nearest_target = target

            if nearest_target:
                # 简单路径检查：直线方向上的位置
                # 在实际游戏中，应该使用更复杂的路径查找
                dir_row = 1 if nearest_target[0] > opp_pos[0] else -1 if nearest_target[0] < opp_pos[0] else 0
                dir_col = 1 if nearest_target[1] > opp_pos[1] else -1 if nearest_target[1] < opp_pos[1] else 0

                # 检查路径上的关键位置
                check_pos = opp_pos
                for _ in range(min_dist):
                    check_pos = (check_pos[0] + dir_row, check_pos[1] + dir_col)
                    if check_pos in my_positions:
                        blocking_positions += 1
                        break

        score += blocking_positions * self.weights.BLOCKING_WEIGHT

        # 威胁评估：我方棋子是否在对手棋子前进方向上
        threat_score = 0
        for my_pos in my_positions:
            for opp_pos in opp_positions:
                if opp_pos in opp_target:
                    continue

                # 简单威胁判断：在我方棋子与对手目标之间的连线上
                dist = self._hex_distance(my_pos, opp_pos)
                if dist <= 3:  # 近距离威胁
                    threat_score += 0.1

        score += threat_score * self.weights.THREAT_WEIGHT

        return score

    def _evaluate_mobility(self, my_positions: List[Tuple], opp_positions: List[Tuple],
                           board_state: Dict) -> float:
        """
        评估移动性

        棋子的潜在移动能力和跳跃机会
        """
        score = 0.0

        # 估算我方棋子的移动自由度
        my_mobility = 0
        for pos in my_positions:
            # 检查单步移动可能性
            for dx, dy in self.directions:
                neighbor = (pos[0] + dx, pos[1] + dy)
                if self._is_valid_position(neighbor) and neighbor not in board_state:
                    my_mobility += 1

            # 检查跳跃可能性（需要更复杂的跳跃逻辑）
            # 这里简化处理

        # 估算对方棋子的移动自由度
        opp_mobility = 0
        for pos in opp_positions:
            for dx, dy in self.directions:
                neighbor = (pos[0] + dx, pos[1] + dy)
                if self._is_valid_position(neighbor) and neighbor not in board_state:
                    opp_mobility += 1

        # 移动性得分：我方移动性 - 对方移动性
        if len(my_positions) > 0 and len(opp_positions) > 0:
            avg_my_mobility = my_mobility / len(my_positions)
            avg_opp_mobility = opp_mobility / len(opp_positions)
            score += (avg_my_mobility - avg_opp_mobility) * self.weights.MOBILITY_WEIGHT

        return score

    def _check_bridge_formation(self, positions: List[Tuple]) -> float:
        """
        检查是否形成桥式结构

        桥式结构便于连续跳跃
        """
        if len(positions) < 3:
            return 0.0

        bridge_score = 0.0

        # 检查是否存在三角形的棋子布局（形成跳跃支点）
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                for k in range(j + 1, len(positions)):
                    # 计算两两距离
                    d12 = self._hex_distance(positions[i], positions[j])
                    d23 = self._hex_distance(positions[j], positions[k])
                    d13 = self._hex_distance(positions[i], positions[k])

                    # 如果三个棋子形成等边三角形（边长2-3），可能是好的桥结构
                    if 2 <= d12 <= 3 and 2 <= d23 <= 3 and 2 <= d13 <= 3:
                        bridge_score += 0.2

        return min(bridge_score, 1.0)  # 限制最大得分

    def _evaluate_clustering(self, positions: List[Tuple]) -> float:
        """
        评估棋子聚类程度

        过度聚类会阻塞移动，适度分散更好
        """
        if len(positions) < 2:
            return 0.0

        # 计算平均最近邻距离
        nearest_distances = []
        for i, pos1 in enumerate(positions):
            min_dist = float('inf')
            for j, pos2 in enumerate(positions):
                if i != j:
                    dist = self._hex_distance(pos1, pos2)
                    if dist < min_dist:
                        min_dist = dist
            nearest_distances.append(min_dist)

        avg_nearest_dist = sum(nearest_distances) / len(nearest_distances)

        # 聚类得分：距离太小表示过度聚类
        if avg_nearest_dist < 2:
            return 1.0  # 高度聚类（负价值）
        elif avg_nearest_dist < 4:
            return 0.5  # 适度聚类
        else:
            return 0.0  # 分散良好

    def _hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        计算六边形网格上的曼哈顿距离（轴向坐标）

        Args:
            pos1: 位置1 (row, col)
            pos2: 位置2 (row, col)

        Returns:
            六边形距离
        """
        r1, q1 = pos1
        r2, q2 = pos2

        # 转换为立方体坐标
        x1 = q1
        z1 = r1
        y1 = -x1 - z1

        x2 = q2
        z2 = r2
        y2 = -x2 - z2

        # 计算立方体坐标下的曼哈顿距离
        return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        检查位置是否在有效棋盘范围内

        Args:
            pos: 位置 (row, col)

        Returns:
            是否有效
        """
        r, c = pos
        # 简化检查：在棋盘边界内
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def quick_evaluate(self, board_state: Dict, current_player: Player) -> float:
        """
        快速评估函数

        用于搜索树的浅层评估，计算速度更快
        """
        # 提取棋子位置
        player1_positions = [pos for pos, player in board_state.items()
                             if player == Player.PLAYER1.value]
        player2_positions = [pos for pos, player in board_state.items()
                             if player == Player.PLAYER2.value]

        if current_player == Player.PLAYER1:
            my_positions = player1_positions
            opp_positions = player2_positions
            my_target = self.player1_target
            opp_target = self.player2_target
        else:
            my_positions = player2_positions
            opp_positions = player1_positions
            my_target = self.player2_target
            opp_target = self.player1_target

        # 快速评估只考虑距离和进度
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

    def _average_distance_to_target(self, positions: List[Tuple], target: set) -> float:
        """计算到目标区域的平均距离"""
        if not positions:
            return 0.0

        total_dist = 0
        for pos in positions:
            if pos in target:
                continue
            min_dist = min(self._hex_distance(pos, t) for t in target)
            total_dist += min_dist

        return total_dist / len(positions)


# 使用示例
if __name__ == "__main__":
    # 创建评估器
    evaluator = ChineseCheckersEvaluator(board_size=17)

    # 示例棋盘状态
    # 注意：这里使用简化表示，实际需要根据棋盘坐标系统
    example_board = {
        (0, 0): Player.PLAYER1.value,  # 玩家1的棋子
        (1, 0): Player.PLAYER1.value,
        (0, 1): Player.PLAYER1.value,
        (16, 16): Player.PLAYER2.value,  # 玩家2的棋子
        (15, 16): Player.PLAYER2.value,
        (16, 15): Player.PLAYER2.value,
        # ... 更多棋子
    }

    # 评估当前局面（从玩家1的角度）
    score = evaluator.evaluate(example_board, Player.PLAYER1)
    print(f"玩家1的评估分数: {score:.2f}")

    # 快速评估
    quick_score = evaluator.quick_evaluate(example_board, Player.PLAYER1)
    print(f"玩家1的快速评估分数: {quick_score:.2f}")