# src/ai/search.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import math
from collections import deque
from src.core.board import ChineseCheckersBoard, CubeCoord
from src.core.moves import ChineseCheckersMoves
from .evaluator import ChineseCheckersEvaluator

class Search:
    def __init__(self, player, depth=3):
        self.player = player  
        self.depth = depth
        
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.eva = None
        
        self.MAX_BRANCH_FACTOR = 8       # 每层最大分支数（确定节点数的关键）
        self.EARLY_BRANCH_FACTOR = 12    # 前几层稍多的分支数
        
        self.ENDGAME_THRESHOLD = 7       # 进入终局的棋子数阈值
        self.BFS_MAX_DEPTH = 8           # BFS最大搜索深度
        self.BFS_MAX_BRANCHES = 8        # BFS每层最大分支
        
        if player == 1:
            self.target_region = 'tri3'  # 玩家1的目标区域
        else:
            self.target_region = 'tri0'  # 玩家2的目标区域
    
    def to_board_player(self, player):
        """将玩家编号(1/2)转换为棋盘表示(1/-1)"""
        return 1 if player == 1 else -1
    
    def get_opponent(self, player):
        """获取对手编号"""
        return 2 if player == 1 else 1
    
    def is_in_endgame(self, board, player):
        """检查是否进入终局阶段"""
        board_player = self.to_board_player(player)  # AI: -1
        opponent_board_player = -board_player       # 对手: 1
        
        # 获取双方棋子
        ai_pieces = board.get_player_pieces(board_player)
        opponent_pieces = board.get_player_pieces(opponent_board_player)
        
        if not ai_pieces or not opponent_pieces:
            return False
        
        # 条件1: 检查最小距离是否大于2
        min_distance = float('inf')
        for ai_piece in ai_pieces:
            for opp_piece in opponent_pieces:
                distance = ai_piece.distance(opp_piece)
                if distance < min_distance:
                    min_distance = distance
                    if min_distance <= self.min_distance_threshold:
                        # 有棋子距离≤2，不满足终局条件
                        return False
        
        if min_distance <= self.min_distance_threshold:
            return False
        
        # 条件2: AI方是否有棋子进入对方坑位
        # AI的目标区域是tri0（东），对手的目标区域是tri3（西）
        # AI的棋子进入对方坑位 = AI棋子在对面的起始区域tri3
        opponent_target_for_ai = 'tri3'  # 对于AI（玩家2）来说，对方的坑位是tri3
        
        ai_pieces_in_opponent_target = 0
        for piece in ai_pieces:
            if board.get_region(piece) == opponent_target_for_ai:
                ai_pieces_in_opponent_target += 1
        
        return ai_pieces_in_opponent_target > 0
    
    def get_pieces_in_target(self, board, player):
        """获取在目标区域的棋子数"""
        board_player = self.to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        in_target = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                in_target += 1
        
        return in_target
    
    def check_winner(self, board, player):
        """检查玩家是否获胜"""
        board_player = self.to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        if len(player_pieces) != 10:
            return False
        
        for coord in player_pieces:
            if board.get_region(coord) != target_region:
                return False
        
        return True
    
    # ==================== 主要接口函数 ====================
    
    def make_move(self, game_state):
        """选择最佳移动 - 主接口"""
        print(f"AI 玩家{self.player} 开始搜索 (深度={self.depth})...")
        
        # 初始化评估器
        self.eva = ChineseCheckersEvaluator(game_state['board'])
        
        board = game_state['board']
        current_player = game_state['current_player']
        
        # 重置统计
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        all_moves = game_state['valid_moves']
        
        if not all_moves:
            return None
            
        if len(all_moves) == 1:
            return all_moves[0]
        
        # 检查立即获胜
        win_move = self._find_immediate_win(board, current_player)
        if win_move:
            print("找到立即获胜的移动")
            return win_move
        
        # 检查是否进入终局阶段
        is_endgame = self.is_in_endgame(board, current_player)
        pieces_in_target = self.get_pieces_in_target(board, current_player)
        
        if is_endgame:
            print(f"进入终局阶段！当前有{pieces_in_target}/10个棋子在目标区域")
            print("使用BFS搜索最优填坑路径...")
            
            # 使用BFS搜索终局最优路径
            best_move = self._endgame_bfs_search(board, current_player, all_moves)
            if best_move:
                print(f"BFS找到终局最优移动")
                return best_move
            else:
                print("BFS搜索失败，回退到Alpha-Beta搜索")
        
        # 使用Alpha-Beta搜索
        print("使用Alpha-Beta搜索...")
        best_move = self._deterministic_alpha_beta_search(board, current_player, all_moves)
        
        return best_move
    
    # ==================== 终局BFS搜索 ====================
    
    def _endgame_bfs_search(self, board, player, all_moves):
        """
        BFS广度优先搜索终局最优路径
        专注于找到最短的进入目标区域的路径
        """
        board_player = self.to_board_player(player)
        target_region = self.target_region
        
        # 1. 首先检查是否有直接进入目标区域的移动
        print("检查直接进入目标区域的移动...")
        for move in all_moves:
            end_pos = move[-1]
            if board.get_region(end_pos) == target_region:
                print(f"找到直接进入目标区域的移动: {move[0]} -> {end_pos}")
                return move
        
        # 2. 如果没有直接进入的移动，使用BFS搜索最短路径
        print("使用BFS搜索最短路径...")
        
        # 获取所有不在目标区域的棋子
        player_pieces = board.get_player_pieces(board_player)
        pieces_outside = []
        for piece in player_pieces:
            if board.get_region(piece) != target_region:
                pieces_outside.append(piece)
        
        if not pieces_outside:
            return all_moves[0] if all_moves else None
        
        print(f"有 {len(pieces_outside)} 个棋子需要移动到目标区域 {target_region}")
        
        # 对每个棋子分别进行BFS搜索
        best_move_for_pieces = []
        
        for piece in pieces_outside:
            print(f"为棋子 {piece} 搜索最短路径...")
            shortest_path = self._bfs_shortest_path_to_target(board, piece, board_player, target_region)
            
            if shortest_path:
                # 将路径转换为移动
                path_move = self._convert_path_to_move(board, piece, shortest_path, board_player)
                if path_move:
                    # 评估路径的质量
                    path_score = self._evaluate_path(board, path_move, player)
                    best_move_for_pieces.append((path_score, path_move))
                    print(f"  找到路径: {len(shortest_path)}步, 得分: {path_score:.1f}")
        
        # 选择最佳的路径
        if best_move_for_pieces:
            best_move_for_pieces.sort(reverse=True, key=lambda x: x[0])
            best_score, best_move = best_move_for_pieces[0]
            print(f"选择最佳路径: {best_move[0]} -> {best_move[-1]}, 得分: {best_score:.1f}")
            return best_move
        
        # 3. 如果BFS找不到路径，使用启发式方法
        print("BFS未找到路径，使用启发式方法...")
        return self._heuristic_endgame_move(board, player, all_moves)
    
    def _bfs_shortest_path_to_target(self, board, start_piece, board_player, target_region):
        """
        使用BFS搜索从棋子到目标区域的最短路径
        返回路径位置列表（包括起点和终点）
        """
        # 使用BFS队列：(当前位置, 路径, 已访问位置集合)
        queue = deque()
        visited = set()
        
        # 初始化队列
        queue.append((start_piece, [start_piece], {start_piece}))
        visited.add(start_piece)
        
        best_path = None
        best_length = float('inf')
        
        depth = 0
        while queue and depth < self.BFS_MAX_DEPTH:
            level_size = len(queue)
            
            for _ in range(level_size):
                current_pos, path, path_visited = queue.popleft()
                
                # 如果当前位置已经在目标区域，更新最佳路径
                if board.get_region(current_pos) == target_region:
                    path_length = len(path) - 1  # 减去起点
                    if path_length < best_length:
                        best_length = path_length
                        best_path = path
                    continue
                
                # 生成当前棋子的所有可能移动
                next_moves = self._get_moves_for_position(board, current_pos, board_player, path_visited)
                
                for move in next_moves[:self.BFS_MAX_BRANCHES]:  # 限制分支
                    next_pos = move[-1]
                    
                    if next_pos not in visited:
                        new_path = path + [next_pos]
                        new_visited = path_visited.copy()
                        new_visited.add(next_pos)
                        
                        visited.add(next_pos)
                        queue.append((next_pos, new_path, new_visited))
            
            depth += 1
        
        return best_path
    
    def _get_moves_for_position(self, board, position, board_player, visited):
        """
        获取某个位置的所有可能移动（单步）
        """
        moves = []
        
        # 1. 单步移动
        for direction in range(6):
            neighbor = position.neighbor(direction)
            if (board.is_valid_cell(neighbor) and 
                board.is_empty(neighbor) and 
                neighbor not in visited):
                moves.append([position, neighbor])
        
        # 2. 跳跃移动（单次跳跃）
        # 这里我们需要检查所有可能的跳跃
        # 使用现有的移动生成器获取所有跳跃
        all_jumps = ChineseCheckersMoves.generate_all_moves(board, board_player)
        for jump in all_jumps:
            if jump[0] == position and jump[-1] not in visited:
                # 检查是否是有效的跳跃（不是连续跳跃）
                if len(jump) == 2:  # 单次跳跃
                    moves.append(jump)
                elif len(jump) > 2:  # 连续跳跃，只取第一步
                    first_jump = [jump[0], jump[1]]
                    if jump[1] not in visited:
                        moves.append(first_jump)
        
        # 3. 按启发式排序移动
        moves.sort(key=lambda m: self._heuristic_distance(m[-1], board, self.target_region))
        
        return moves
    
    def _heuristic_distance(self, position, board, target_region):
        """
        计算位置到目标区域的启发式距离
        值越小表示越接近目标
        """
        # 如果已经在目标区域，距离为0
        if board.get_region(position) == target_region:
            return 0
        
        # 基于区域距离的启发式
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        current_region = board.get_region(position)
        target_idx = region_order.index(target_region)
        
        try:
            current_idx = region_order.index(current_region)
            return abs(current_idx - target_idx)
        except ValueError:
            return 10  # 默认较大的距离
    
    def _convert_path_to_move(self, board, start_piece, path, board_player):
        """
        将路径位置列表转换为实际的移动序列
        """
        if not path or len(path) < 2:
            return None
        
        # 检查是否存在直接跳跃连接
        # 首先检查从起点到终点的直接跳跃
        start = path[0]
        end = path[-1]
        
        # 获取所有可能的移动
        all_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        # 查找从起点到终点的移动
        for move in all_moves:
            if move[0] == start and move[-1] == end:
                return move
        
        # 如果没有直接移动，构造逐步移动
        # 这里简化处理：只取第一步
        if len(path) >= 2:
            first_step = path[1]
            # 查找从起点到第一步的移动
            for move in all_moves:
                if move[0] == start and move[-1] == first_step:
                    return move
        
        # 如果找不到，返回空
        return None
    
    def _evaluate_path(self, board, move, player):
        """
        评估路径的质量
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self.to_board_player(player)
        target_region = self.target_region
        
        # 1. 是否进入目标区域（最高奖励）
        if board.get_region(end) == target_region:
            score += 1000
        
        # 2. 路径长度（越短越好）
        path_length = len(move) - 1  # 跳跃次数
        if path_length == 1:
            score += 100  # 单步移动奖励
        else:
            score += max(0, 200 - path_length * 20)  # 路径越长奖励越少
        
        # 3. 前进方向
        start_region = board.get_region(start)
        end_region = board.get_region(end)
        
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        target_idx = region_order.index(target_region)
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            
            start_dist = abs(start_idx - target_idx)
            end_dist = abs(end_idx - target_idx)
            
            if end_dist < start_dist:  # 向目标前进
                score += 50 * (start_dist - end_dist)
        except ValueError:
            pass
        
        return score
    
    def _heuristic_endgame_move(self, board, player, all_moves):
        """
        启发式选择终局移动
        当BFS找不到路径时使用
        """
        board_player = self.to_board_player(player)
        target_region = self.target_region
        
        best_move = None
        best_score = -float('inf')
        
        for move in all_moves[:20]:  # 只检查前20个移动
            score = self._evaluate_simple_endgame_move(board, move, player)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _evaluate_simple_endgame_move(self, board, move, player):
        """
        简单评估终局移动
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self.to_board_player(player)
        target_region = self.target_region
        
        # 1. 是否进入目标区域
        if board.get_region(end) == target_region:
            score += 500
        
        # 2. 距离目标区域的变化
        start_region = board.get_region(start)
        end_region = board.get_region(end)
        
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        target_idx = region_order.index(target_region)
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            
            start_dist = abs(start_idx - target_idx)
            end_dist = abs(end_idx - target_idx)
            
            if end_dist < start_dist:  # 向目标前进
                score += 30 * (start_dist - end_dist)
        except ValueError:
            pass
        
        # 3. 移动效率
        jump_count = len(move) - 1
        score += max(0, 50 - jump_count * 5)  # 跳跃次数越少越好
        
        return score
    
    # ==================== Alpha-Beta搜索核心 ====================
    
    def _deterministic_alpha_beta_search(self, board, player, all_moves):
        """
        确定性的Alpha-Beta搜索
        """
        # 第1步：对根节点的移动进行贪心排序
        sorted_root_moves = self._greedy_sort_moves(board, all_moves, player, 0)
        
        # 限制根节点的分支数
        max_root_branches = min(self.EARLY_BRANCH_FACTOR, len(sorted_root_moves))
        root_moves = sorted_root_moves[:max_root_branches]
        
        print(f"Alpha-Beta搜索: 深度={self.depth}, 根分支={len(root_moves)}")
        
        # Alpha-Beta搜索参数
        alpha = -float('inf')
        beta = float('inf')
        best_score = -float('inf')
        best_move = root_moves[0] if root_moves else None
        
        # 第2步：对每个候选移动进行Alpha-Beta搜索
        for i, move in enumerate(root_moves):
            new_board = ChineseCheckersMoves.apply_move(board, move)
            
            if player == self.player:
                score = self._alpha_beta_min(new_board, self.depth-1, alpha, beta, 
                                           self.get_opponent(player))
            else:
                score = self._alpha_beta_max(new_board, self.depth-1, alpha, beta, 
                                           self.get_opponent(player))
            
            # 更新最佳移动
            if score > best_score:
                best_score = score
                best_move = move
            
            # Alpha-Beta剪枝
            if player == self.player:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        # 输出统计信息
        total_possible_nodes = self._calculate_possible_nodes(self.depth)
        efficiency = self.nodes_evaluated / total_possible_nodes if total_possible_nodes > 0 else 0
        
        print(f"搜索统计:")
        print(f"  评估节点数: {self.nodes_evaluated}")
        print(f"  剪枝次数: {self.pruning_count}")
        print(f"  搜索效率: {efficiency:.1%}")
        
        return best_move
    
    def _alpha_beta_max(self, board, depth, alpha, beta, player):
        """Max层搜索"""
        self.nodes_evaluated += 1
        
        if depth == 0:
            return self._evaluate_board(board, player)
        
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        board_player = self.to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self._evaluate_board(board, player)
        
        # 贪心排序并限制分支数
        sorted_moves = self._greedy_sort_moves(board, valid_moves, player, depth)
        max_branches = self._get_branch_factor(depth)
        moves_to_search = sorted_moves[:max_branches]
        
        max_eval = -float('inf')
        
        for move in moves_to_search:
            new_board = ChineseCheckersMoves.apply_move(board, move)
            next_player = self.get_opponent(player)
            eval_score = self._alpha_beta_min(new_board, depth-1, alpha, beta, next_player)
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        return max_eval
    
    def _alpha_beta_min(self, board, depth, alpha, beta, player):
        """Min层搜索"""
        self.nodes_evaluated += 1
        
        if depth == 0:
            return self._evaluate_board(board, player)
        
        if self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        elif self.check_winner(board, self.player):
            return 10000 + depth * 100
        
        board_player = self.to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self._evaluate_board(board, player)
        
        sorted_moves = self._greedy_sort_moves(board, valid_moves, player, depth)
        max_branches = self._get_branch_factor(depth)
        moves_to_search = sorted_moves[:max_branches]
        
        min_eval = float('inf')
        
        for move in moves_to_search:
            new_board = ChineseCheckersMoves.apply_move(board, move)
            next_player = self.get_opponent(player)
            eval_score = self._alpha_beta_max(new_board, depth-1, alpha, beta, next_player)
            
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        return min_eval
    
    # ==================== 贪心排序函数 ====================
    
    def _greedy_sort_moves(self, board, moves, player, depth):
        """贪心排序移动"""
        if not moves:
            return []
        
        scored_moves = []
        for move in moves:
            score = self._quick_move_evaluation(board, move, player, depth)
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]
    
    def _quick_move_evaluation(self, board, move, player, depth):
        """快速评估移动质量"""
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self.to_board_player(player)
        target_region = board.player_target_regions[board_player]
        end_region = board.get_region(end)
        start_region = board.get_region(start)
        
        # 1. 目标区域奖励
        if end_region == target_region:
            score += 2000
        
        # 2. 前进方向奖励
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        target_name = 'tri3' if player == 1 else 'tri0'
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            target_idx = region_order.index(target_name)
            
            start_dist = abs(start_idx - target_idx)
            end_dist = abs(end_idx - target_idx)
            
            if end_dist < start_dist:
                score += 300 * (start_dist - end_dist)
            elif end_dist > start_dist:
                score -= 500
                
            if depth > 0:
                score *= 0.9
        except ValueError:
            pass
        
        # 3. 跳跃奖励
        jump_count = len(move) - 1
        if jump_count > 1:
            score += 80 * jump_count
        elif not start.is_neighbor(end):
            score += 50
        
        return score
    
    def _evaluate_board(self, board, player):
        """评估棋盘状态"""
        if self.eva is None:
            self.eva = ChineseCheckersEvaluator(board)
        
        board_player = self.to_board_player(player)
        return self.eva.evaluate(board.cells.copy(), board_player)
    
    # ==================== 辅助函数 ====================
    
    def _find_immediate_win(self, board, player):
        """查找立即获胜的移动"""
        board_player = self.to_board_player(player)
        moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        for move in moves:
            test_board = ChineseCheckersMoves.apply_move(board, move)
            if self.check_winner(test_board, player):
                return move
        return None
    
    def _get_branch_factor(self, depth):
        """根据深度获取分支因子"""
        if depth >= self.depth - 1:
            return self.EARLY_BRANCH_FACTOR
        else:
            return self.MAX_BRANCH_FACTOR
    
    def _calculate_possible_nodes(self, depth):
        """计算理论最大节点数"""
        if depth <= 0:
            return 1
        
        total_nodes = self.EARLY_BRANCH_FACTOR
        
        for d in range(1, depth):
            if d <= 1:
                total_nodes *= self.EARLY_BRANCH_FACTOR
            else:
                total_nodes *= self.MAX_BRANCH_FACTOR
        
        return total_nodes