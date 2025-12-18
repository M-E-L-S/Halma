# src/ai/search.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import math
from src.core.board import ChineseCheckersBoard, CubeCoord
from src.core.moves import ChineseCheckersMoves
from .evaluator import ChineseCheckersEvaluator

class Search:
    def __init__(self, player, depth=3):
        self.player = player  # AI控制的玩家编号：2
        self.depth = depth
        
        # 搜索统计
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.eva = None
        
        # 搜索参数
        self.max_branch_factor = 20
        self.early_move_factor = 12
        
        # 终局搜索参数
        self.endgame_threshold = 7  # 当有7个以上棋子在目标区域时，进入终局模式
        self.dfs_depth_limit = 8    # DFS搜索深度限制
        self.endgame_cache = {}     # 终局搜索缓存
    
    def to_board_player(self, player):
        return 1 if player == 1 else -1
    
    def get_opponent(self, player):
        return 2 if player == 1 else 1
    
    def get_target_region_name(self, player):
        return 'tri3' if player == 1 else 'tri0'
    
    def is_in_endgame(self, board, player):
        """检查是否进入终局阶段"""
        board_player = self.to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        # 计算在目标区域的棋子数
        in_target = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                in_target += 1
        
        return in_target >= self.endgame_threshold
    
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
    
    def find_immediate_win(self, board, player):
        """查找立即获胜的移动"""
        board_player = self.to_board_player(player)
        moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        for move in moves:
            test_board = ChineseCheckersMoves.apply_move(board, move)
            if self.check_winner(test_board, player):
                return move
        return None
    
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
    
    def make_move(self, game_state):
        """选择最佳移动 - 主接口"""
        print(f"AI 玩家{self.player} 开始思考...")
        
        # 初始化评估器
        self.eva = ChineseCheckersEvaluator(game_state['board'])
        
        board = game_state['board']
        current_player = game_state['current_player']
        
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        all_moves = game_state['valid_moves']
        
        if not all_moves:
            return None
            
        if len(all_moves) == 1:
            return all_moves[0]
        
        # 检查立即获胜
        win_move = self.find_immediate_win(board, current_player)
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
            best_move = self._endgame_dfs_search(board, current_player)
            if best_move:
                print(f"DFS找到终局移动: {best_move}")
                return best_move
            else:
                print("DFS搜索失败，回退到Alpha-Beta搜索")
        
        # 使用Alpha-Beta搜索（中局策略）
        print("使用Alpha-Beta搜索中局策略...")
        best_move = self._midgame_alpha_beta_search(board, current_player, all_moves)
        
        return best_move
    
    def _endgame_dfs_search(self, board, player):
        """
        DFS搜索终局最优路径
        专注于将剩余棋子移动到目标区域
        """
        board_player = self.to_board_player(player)
        target_region = self.get_target_region_name(player)
        
        # 获取所有不在目标区域的棋子
        player_pieces = board.get_player_pieces(board_player)
        pieces_outside = []
        for piece in player_pieces:
            if board.get_region(piece) != target_region:
                pieces_outside.append(piece)
        
        if not pieces_outside:
            # 所有棋子都在目标区域，随机移动
            moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
            return moves[0] if moves else None
        
        print(f"有{len(pieces_outside)}个棋子需要移动到目标区域{target_region}")
        
        # 对每个不在目标区域的棋子，搜索到达目标区域的最短路径
        best_overall_move = None
        best_overall_score = -float('inf')
        
        for piece in pieces_outside:
            # 搜索从该棋子到目标区域的最优路径
            piece_move = self._find_best_path_to_target(board, piece, target_region, board_player)
            if piece_move:
                # 评估这个移动的质量
                test_board = ChineseCheckersMoves.apply_move(board, piece_move)
                score = self._evaluate_endgame_move(board, test_board, player, piece_move)
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_move = piece_move
        
        return best_overall_move
    
    def _find_best_path_to_target(self, board, start_piece, target_region, board_player):
        """
        使用BFS/DFS找到从棋子到目标区域的最短路径
        """
        from collections import deque
        
        # 获取目标区域的所有空位
        target_cells = []
        for coord, region in board.regions.items():
            if region == target_region and board.is_empty(coord):
                target_cells.append(coord)
        
        if not target_cells:
            return None
        
        # 使用BFS搜索最短路径
        queue = deque()
        visited = set()
        
        # (当前位置, 路径, 已访问位置集合)
        queue.append((start_piece, [start_piece], {start_piece}))
        visited.add(start_piece)
        
        best_path = None
        best_length = float('inf')
        
        while queue:
            current_pos, path, path_visited = queue.popleft()
            
            # 如果已经在目标区域，检查是否是最短路径
            if board.get_region(current_pos) == target_region:
                if len(path) < best_length:
                    best_length = len(path)
                    best_path = path
                continue
            
            # 限制搜索深度
            if len(path) > self.dfs_depth_limit:
                continue
            
            # 生成所有可能的移动
            for move in self._generate_moves_for_piece(board, current_pos, board_player, path_visited):
                if len(move) < 2:
                    continue
                    
                next_pos = move[-1]
                if next_pos not in visited:
                    new_path = path + [next_pos]
                    new_visited = path_visited.copy()
                    new_visited.add(next_pos)
                    
                    visited.add(next_pos)
                    queue.append((next_pos, new_path, new_visited))
        
        return best_path if best_path and len(best_path) > 1 else None
    
    def _generate_moves_for_piece(self, board, piece, player, visited):
        """生成单个棋子的所有可能移动（不考虑连续跳跃）"""
        moves = []
        
        # 单步移动
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if (board.is_valid_cell(neighbor) and 
                board.is_empty(neighbor) and 
                neighbor not in visited):
                moves.append([piece, neighbor])
        
        # 单次跳跃（不递归连续跳跃以简化搜索）
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if not board.is_valid_cell(neighbor) or board.is_empty(neighbor):
                continue
            
            # 检查跳跃
            jump_target = piece.neighbor(direction * 2)  # 简化处理
            if (board.is_valid_cell(jump_target) and 
                board.is_empty(jump_target) and 
                jump_target not in visited):
                
                # 检查是否是有效跳跃
                if ChineseCheckersMoves._is_valid_jump(board, piece, jump_target):
                    moves.append([piece, jump_target])
        
        return moves
    
    def _evaluate_endgame_move(self, original_board, new_board, player, move):
        """
        评估终局移动的质量
        专注于：1) 是否进入目标区域 2) 是否让出其他棋子的路径
        """
        score = 0
        board_player = self.to_board_player(player)
        target_region = self.get_target_region_name(player)
        
        start = move[0]
        end = move[-1]
        
        # 1. 是否移动到目标区域（最重要的因素）
        if board.get_region(end) == target_region:
            score += 1000
        
        # 2. 移动距离（越短越好）
        score -= len(move) * 10
        
        # 3. 是否阻挡了其他棋子
        # 检查移动后是否阻挡了其他棋子到达目标区域
        pieces_outside = []
        for piece in new_board.get_player_pieces(board_player):
            if new_board.get_region(piece) != target_region:
                pieces_outside.append(piece)
        
        # 计算阻挡分数
        for piece in pieces_outside:
            if piece != end:  # 排除自己
                # 检查从该棋子到目标区域的路径是否被阻挡
                if self._is_path_blocked(new_board, piece, target_region, board_player):
                    score -= 50
        
        # 4. 是否为其他棋子让出路径
        if original_board.get_region(start) != target_region:
            # 检查离开的位置是否对其他棋子有用
            for other_piece in pieces_outside:
                if other_piece != start:
                    # 检查其他棋子是否能使用这个位置
                    if self._can_use_position(new_board, other_piece, start, target_region, board_player):
                        score += 30
        
        return score
    
    def _is_path_blocked(self, board, piece, target_region, board_player):
        """检查棋子到达目标区域的路径是否被阻挡"""
        # 简化的路径检查：检查是否有直接邻居在目标区域
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if (board.is_valid_cell(neighbor) and 
                board.get_region(neighbor) == target_region and 
                board.is_empty(neighbor)):
                return False
        return True
    
    def _can_use_position(self, board, piece, position, target_region, board_player):
        """检查棋子是否能使用某个位置到达目标区域"""
        # 检查位置是否在目标区域
        if board.get_region(position) == target_region:
            # 检查是否能移动到该位置
            for move in self._generate_moves_for_piece(board, piece, board_player, set()):
                if len(move) > 1 and move[-1] == position:
                    return True
        return False
    
    def _midgame_alpha_beta_search(self, board, current_player, all_moves):
        """中局Alpha-Beta搜索"""
        # 对根节点移动进行贪心排序
        root_moves = self._greedy_order_moves_midgame(board, all_moves, current_player, 0)
        
        alpha = -float('inf')
        beta = float('inf')
        best_score = -float('inf')
        best_move = None
        
        print(f"Alpha-Beta搜索深度: {self.depth}, 移动数: {len(root_moves)}")
        
        for i, move in enumerate(root_moves):
            test_board = ChineseCheckersMoves.apply_move(board, move)
            
            if current_player == self.player:
                score = self._alpha_beta_enhanced(
                    test_board, self.depth - 1, False, alpha, beta, 
                    self.get_opponent(current_player)
                )
            else:
                score = self._alpha_beta_enhanced(
                    test_board, self.depth - 1, True, alpha, beta, current_player
                )
            
            print(f"移动 {i+1}/{len(root_moves)}: {move} 得分: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_move = move
            
            if current_player == self.player:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        print(f"评估节点数: {self.nodes_evaluated}, 剪枝次数: {self.pruning_count}")
        print(f"最佳评估值: {best_score:.2f}")
        
        return best_move if best_move else root_moves[0]
    
    def _greedy_order_moves_midgame(self, board, moves, player, depth):
        """中局贪心排序"""
        if not moves:
            return []
        
        scored_moves = []
        for move in moves:
            score = self._quick_evaluate_midgame_move(board, move, player, depth)
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        if depth > 2:
            max_moves = min(self.early_move_factor, len(scored_moves))
        else:
            max_moves = min(self.max_branch_factor, len(scored_moves))
        
        return [move for _, move in scored_moves[:max_moves]]
    
    def _quick_evaluate_midgame_move(self, board, move, player, depth):
        """中局快速评估"""
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self.to_board_player(player)
        target_region = board.player_target_regions[board_player]
        end_region = board.get_region(end)
        
        # 1. 目标区域奖励
        if end_region == target_region:
            score += 500
        
        # 2. 前进距离
        start_region = board.get_region(start)
        if start_region != end_region:
            region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
            target_region_name = self.get_target_region_name(player)
            
            try:
                start_idx = region_order.index(start_region)
                end_idx = region_order.index(end_region)
                target_idx = region_order.index(target_region_name)
                
                start_dist = abs(start_idx - target_idx)
                end_dist = abs(end_idx - target_idx)
                
                if end_dist < start_dist:
                    score += 150 * (start_dist - end_dist)
                elif end_dist > start_dist:
                    score -= 200
            except ValueError:
                pass
        
        # 3. 跳跃奖励
        if len(move) > 2:
            jump_bonus = 40 if depth > 1 else 60
            score += jump_bonus * (len(move) - 1)
        elif not start.is_neighbor(end):
            jump_bonus = 25 if depth > 1 else 40
            score += jump_bonus
        
        # 4. 中心区域奖励
        if end_region == 'hex':
            score += 30
        
        # 5. 防御性考虑（中局重要）
        opponent = self.get_opponent(player)
        if depth <= 1:
            opponent_moves = ChineseCheckersMoves.generate_all_moves(
                board, self.to_board_player(opponent)
            )
            for opp_move in opponent_moves[:3]:
                opp_end = opp_move[-1]
                if opp_end == end:
                    score += 80  # 阻挡对手
        
        return score
    
    def _alpha_beta_enhanced(self, board, depth, is_maximizing, alpha, beta, player):
        """Alpha-Beta搜索核心"""
        self.nodes_evaluated += 1
        
        # 终止条件
        if depth == 0:
            return self.evaluate_midgame(board, player)
        
        # 检查游戏是否结束
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        # 生成移动
        board_player = self.to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self.evaluate_midgame(board, player)
        
        # 排序移动
        ordered_moves = self._greedy_order_moves_midgame(board, valid_moves, player, depth)
        
        if is_maximizing:
            max_eval = -float('inf')
            
            for move in ordered_moves:
                test_board = ChineseCheckersMoves.apply_move(board, move)
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta_enhanced(
                    test_board, depth - 1, False, alpha, beta, next_player
                )
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            
            for move in ordered_moves:
                test_board = ChineseCheckersMoves.apply_move(board, move)
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta_enhanced(
                    test_board, depth - 1, True, alpha, beta, next_player
                )
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            return min_eval
    
    def evaluate_midgame(self, board, player):
        """中局评估函数"""
        if self.eva is None:
            self.eva = ChineseCheckersEvaluator(board)
        
        board_player = self.to_board_player(player)
        return self.eva.evaluate(board.cells.copy(), board_player)