# src/ai/search.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import math
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
        self.DFS_MAX_DEPTH = 8           # DFS最大搜索深度
        self.DFS_MAX_BRANCHES = 6        # DFS每层最大分支
        self.endgame_cache = {}          # 终局搜索缓存
        
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
        board_player = self.to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        in_target = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                in_target += 1
        
        return in_target >= self.ENDGAME_THRESHOLD
    
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
            print("使用DFS搜索最优填坑路径...")
            
            # 使用DFS搜索终局最优路径
            best_move = self._endgame_dfs_search(board, current_player, all_moves)
            if best_move:
                print(f"DFS找到终局最优移动")
                return best_move
            else:
                print("DFS搜索失败，回退到Alpha-Beta搜索")
        
        # 使用Alpha-Beta搜索
        print("使用Alpha-Beta搜索...")
        best_move = self._deterministic_alpha_beta_search(board, current_player, all_moves)
        
        return best_move
    
    
    def _endgame_dfs_search(self, board, player, all_moves):
        """
        DFS深度优先搜索终局最优路径
        专注于将剩余棋子移动到目标区域
        """
        board_player = self.to_board_player(player)
        target_region = self.target_region
        
        # 1. 分析当前状态
        player_pieces = board.get_player_pieces(board_player)
        
        # 获取所有不在目标区域的棋子
        pieces_outside = []
        for piece in player_pieces:
            if board.get_region(piece) != target_region:
                pieces_outside.append(piece)
        
        if not pieces_outside:
            return all_moves[0] if all_moves else None
        
        print(f"有 {len(pieces_outside)} 个棋子需要移动到目标区域 {target_region}")
        
        # 2. 对每个不在目标区域的棋子，计算到达目标区域的潜力
        best_move = None
        best_score = -float('inf')
        
        # 尝试每个棋子的可能移动
        for piece in pieces_outside:
            piece_moves = self._get_moves_for_piece(board, piece, board_player)
            
            for move in piece_moves[:self.DFS_MAX_BRANCHES]:
                new_board = ChineseCheckersMoves.apply_move(board, move)
                
                # 评估这个移动的终局价值
                score = self._evaluate_endgame_move(board, new_board, move, player)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                    # 如果这个移动直接进入目标区域，优先考虑
                    end_pos = move[-1]
                    if board.get_region(end_pos) == target_region:
                        print(f"找到直接进入目标区域的移动: {move[0]} -> {end_pos}")
                        return move
        
        # 3. 如果没有找到直接进入目标区域的移动，尝试多步DFS
        if best_move and best_score < 1000:  # 如果没有找到很好的移动
            print("尝试深度DFS搜索...")
            dfs_move = self._deep_dfs_search(board, player, pieces_outside, depth=0)
            if dfs_move:
                return dfs_move
        
        return best_move
    
    def _deep_dfs_search(self, board, player, pieces_outside, depth=0):
        """
        深度DFS搜索多步最优路径
        """
        if depth >= self.DFS_MAX_DEPTH:
            return None
        
        board_player = self.to_board_player(player)
        target_region = self.target_region
        
        best_sequence = None
        best_sequence_score = -float('inf')
        
        # 尝试每个棋子的可能移动
        for piece in pieces_outside:
            piece_moves = self._get_moves_for_piece(board, piece, board_player)
            
            for move in piece_moves[:self.DFS_MAX_BRANCHES]:
                new_board = ChineseCheckersMoves.apply_move(board, move)
                
                # 如果这个移动进入目标区域，直接返回
                if board.get_region(move[-1]) == target_region:
                    print(f"深度{depth}: 找到进入目标区域的移动")
                    return move
                
                # 递归搜索下一步
                new_pieces_outside = []
                for p in pieces_outside:
                    if p != piece:  # 移除已经移动的棋子
                        new_pieces_outside.append(p)
                
                # 如果还有其他棋子需要移动，继续搜索
                if new_pieces_outside:
                    next_move = self._deep_dfs_search(new_board, player, new_pieces_outside, depth+1)
                    if next_move:
                        # 计算整个序列的得分
                        full_move = move + next_move[1:] if len(next_move) > 1 else move
                        sequence_score = self._evaluate_move_sequence(board, full_move, player)
                        
                        if sequence_score > best_sequence_score:
                            best_sequence_score = sequence_score
                            best_sequence = full_move
        
        return best_sequence
    
    def _evaluate_move_sequence(self, board, move_sequence, player):
        """
        评估移动序列的终局价值
        """
        if not move_sequence or len(move_sequence) < 2:
            return 0
        
        total_score = 0
        current_board = board.copy()
        
        for i in range(len(move_sequence) - 1):
            # 假设move_sequence是一系列位置
            if i == 0:
                start = move_sequence[i]
                end = move_sequence[i+1]
                move = [start, end]
            else:
                # 对于多步序列，需要重构移动
                continue
            
            new_board = ChineseCheckersMoves.apply_move(current_board, move)
            step_score = self._evaluate_endgame_move(current_board, new_board, move, player)
            total_score += step_score * (0.9 ** i)  # 越远的步骤权重越低
            
            current_board = new_board
        
        return total_score
    
    def _get_moves_for_piece(self, board, piece, board_player):
        """
        获取单个棋子的所有可能移动
        优化版本：优先考虑进入目标区域的移动
        """
        target_region = self.target_region
        
        # 获取所有合法移动
        all_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        # 过滤出以该棋子为起点的移动
        piece_moves = []
        target_moves = []  # 进入目标区域的移动
        
        for move in all_moves:
            if move[0] == piece:
                # 检查是否进入目标区域
                if board.get_region(move[-1]) == target_region:
                    target_moves.append(move)
                else:
                    piece_moves.append(move)
        
        # 优先返回进入目标区域的移动
        if target_moves:
            # 对目标移动按跳跃次数排序（跳跃次数少优先）
            target_moves.sort(key=lambda m: len(m))
            return target_moves + piece_moves
        
        # 如果没有目标移动，对普通移动排序
        piece_moves.sort(key=lambda m: self._rank_move_for_endgame(board, m))
        return piece_moves
    
    def _rank_move_for_endgame(self, board, move):
        """
        为终局移动排序
        返回越小越好的排名值
        """
        start = move[0]
        end = move[-1]
        target_region = self.target_region
        
        # 基础排名（跳跃次数）
        rank = len(move) * 10
        
        # 如果向目标区域前进，降低排名
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
                rank -= (start_dist - end_dist) * 20
        except ValueError:
            pass
        
        return rank
    
    def _evaluate_endgame_move(self, original_board, new_board, move, player):
        """
        评估终局移动的价值
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self.to_board_player(player)
        target_region = self.target_region
        
        # 1. 是否进入目标区域（最高奖励）
        if original_board.get_region(end) == target_region:
            score += 500
        
        # 2. 距离目标区域的变化
        start_region = original_board.get_region(start)
        end_region = original_board.get_region(end)
        
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
        
        # 3. 移动效率（跳跃次数少更好）
        jump_count = len(move) - 1
        if jump_count == 1:
            score += 10  # 单步移动在终局中可能更好
        else:
            score += 5* jump_count  # 跳跃奖励，但每步奖励较少
        
        # 4. 是否阻挡其他棋子
        # 检查移动后是否阻挡了其他棋子到达目标区域
        pieces_outside = []
        for piece in new_board.get_player_pieces(board_player):
            if new_board.get_region(piece) != target_region:
                pieces_outside.append(piece)
        
        for piece in pieces_outside:
            if piece != end:  # 排除自己
                # 简化的阻挡检查：是否在关键路径上
                if self._is_on_critical_path(new_board, piece, target_region):
                    score -= 10
        
        # 5. 是否为其他棋子创造机会
        if original_board.get_region(start) != target_region:
            # 检查离开的位置是否对其他棋子有用
            for other_piece in pieces_outside:
                if other_piece != start:
                    if self._position_is_useful_for_piece(new_board, other_piece, start, target_region):
                        score += 10
        
        return score
    
    def _is_on_critical_path(self, board, piece, target_region):
        """
        检查棋子是否在关键路径上（简化版本）
        """
        # 检查是否有直接邻居在目标区域
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if (board.is_valid_cell(neighbor) and 
                board.get_region(neighbor) == target_region):
                return True
        
        # 检查是否在通往目标区域的直线路径上
        center = CubeCoord(0, 0, 0)
        if piece.distance(center) <= 2:
            return True
        
        return False
    
    def _position_is_useful_for_piece(self, board, piece, position, target_region):
        """
        检查某个位置对棋子是否有用
        """
        # 位置必须在目标区域
        if board.get_region(position) != target_region:
            return False
        
        # 检查棋子是否能到达该位置
        # 简化的检查：是否相邻
        if piece.is_neighbor(position) and board.is_empty(position):
            return True
        
        # 检查是否有一条跳跃路径
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if (board.is_valid_cell(neighbor) and not board.is_empty(neighbor)):
                jump_target = piece.neighbor(direction * 2)
                if jump_target == position:
                    return True
        
        return False
    
    
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