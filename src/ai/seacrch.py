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
        self.player = player  # AI控制的玩家编号：2
        self.depth = depth
        
        # 搜索统计
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.eva = None
        
        # 搜索参数
        self.max_branch_factor = 20
        self.early_move_factor = 12
        
        # 终局参数
        self.min_distance_threshold = 2  # 最小距离阈值
        self.endgame_cache = {}          # 终局搜索缓存
        self.endgame_move_limit = 15     # 终局移动搜索限制
    
    def to_board_player(self, player):
        return 1 if player == 1 else -1
    
    def get_opponent(self, player):
        return 2 if player == 1 else 1
    
    def get_target_region_name(self, player):
        """获取玩家目标区域名称"""
        # 玩家1（外部1，内部1）目标：tri3（西）
        # 玩家2（外部2，内部-1）目标：tri0（东）
        return 'tri3' if player == 1 else 'tri0'
    
    def is_in_endgame(self, board, player):
        """
        检查是否进入终局阶段
        
        条件：
        1. AI方和对手方棋子的最小距离大于2
        2. AI方有棋子已经进入对方坑位（目标区域）
        """
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
        opponent_target_for_ai = 'tri3'  # 对于AI（玩家2）来说，对方的坑位是tri3
        
        ai_pieces_in_opponent_target = 0
        for piece in ai_pieces:
            if board.get_region(piece) == opponent_target_for_ai:
                ai_pieces_in_opponent_target += 1
        
        return ai_pieces_in_opponent_target > 0
    
    def get_min_distance_between_players(self, board, player):
        """获取双方棋子的最小距离"""
        board_player = self.to_board_player(player)
        opponent_board_player = -board_player
        
        ai_pieces = board.get_player_pieces(board_player)
        opponent_pieces = board.get_player_pieces(opponent_board_player)
        
        if not ai_pieces or not opponent_pieces:
            return float('inf')
        
        min_distance = float('inf')
        for ai_piece in ai_pieces:
            for opp_piece in opponent_pieces:
                distance = ai_piece.distance(opp_piece)
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def count_pieces_in_opponent_target(self, board, player):
        """统计AI方进入对方坑位的棋子数"""
        board_player = self.to_board_player(player)
        opponent_target_for_ai = 'tri3'  # AI的对手坑位
        
        ai_pieces = board.get_player_pieces(board_player)
        count = 0
        for piece in ai_pieces:
            if board.get_region(piece) == opponent_target_for_ai:
                count += 1
        
        return count
    
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
        print(f"AI 玩家{self.player} 开始思考...")
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
        min_distance = self.get_min_distance_between_players(board, current_player)
        pieces_in_opponent_target = self.count_pieces_in_opponent_target(board, current_player)
        
        if is_endgame:
            print(f"进入终局阶段！")
            print(f"双方棋子最小距离: {min_distance} (>{self.min_distance_threshold})")
            print(f"AI有{pieces_in_opponent_target}个棋子进入对方坑位")
            print("使用终局DFS搜索最优填坑路径...")
            # 使用终局DFS搜索
            best_move = self._endgame_dfs_search(board, current_player)
            if best_move:
                print(f"终局DFS找到移动: {best_move}")
                return best_move
            else:
                print("终局DFS搜索失败，回退到Alpha-Beta搜索")
        
        print("使用Alpha-Beta搜索中局策略...")
        best_move = self._midgame_alpha_beta_search(board, current_player, all_moves)
        return best_move
    
    def _endgame_dfs_search(self, board, player):
        """
        终局DFS搜索 - 专注于填满目标区域
        
        策略：
        1. 优先移动已经在对方坑位的棋子到正确位置
        2. 移动其他棋子进入对方坑位
        3. 避免阻挡已方棋子的路径
        """
        board_player = self.to_board_player(player)  # AI: -1
        ai_target_region = 'tri0'  # AI的目标区域（东）
        opponent_target_region = 'tri3'  # 对手的目标区域，也是AI的"对方坑位"
        
        # 获取AI所有棋子
        ai_pieces = board.get_player_pieces(board_player)
        
        # 分类棋子
        pieces_in_ai_target = []      # 在AI自己目标区域的棋子
        pieces_in_opponent_target = [] # 在对方坑位的棋子
        pieces_elsewhere = []         # 在其他位置的棋子
        
        for piece in ai_pieces:
            region = board.get_region(piece)
            if region == ai_target_region:
                pieces_in_ai_target.append(piece)
            elif region == opponent_target_region:
                pieces_in_opponent_target.append(piece)
            else:
                pieces_elsewhere.append(piece)
        
        print(f"棋子分布: AI目标区域={len(pieces_in_ai_target)}, 对方坑位={len(pieces_in_opponent_target)}, 其他位置={len(pieces_elsewhere)}")
        
        # 策略1: 优先处理已经在对方坑位的棋子
        if pieces_in_opponent_target:
            print("优先移动已在对方坑位的棋子...")
            for piece in pieces_in_opponent_target:
                # 寻找从该棋子到AI目标区域（tri0）的路径
                best_move = self._find_path_to_region(board, piece, ai_target_region, board_player)
                if best_move and len(best_move) > 1:
                    # 检查这个移动是否合理
                    if self._is_safe_endgame_move(board, best_move, player):
                        return best_move
        
        # 策略2: 移动其他棋子进入对方坑位
        if pieces_elsewhere:
            print("移动其他棋子进入对方坑位...")
            # 先找可以直接跳进对方坑位的棋子
            for piece in pieces_elsewhere:
                # 寻找直接进入对方坑位的移动
                direct_move = self._find_direct_move_to_region(board, piece, opponent_target_region, board_player)
                if direct_move:
                    if self._is_safe_endgame_move(board, direct_move, player):
                        return direct_move
            
            # 如果没有直接移动，找最短路径
            best_move_for_elsewhere = None
            best_score = -float('inf')
            
            for piece in pieces_elsewhere:
                path_move = self._find_path_to_region(board, piece, opponent_target_region, board_player)
                if path_move and len(path_move) > 1:
                    score = self._evaluate_endgame_path(path_move, piece, opponent_target_region)
                    if score > best_score:
                        best_score = score
                        best_move_for_elsewhere = path_move
            
            if best_move_for_elsewhere and self._is_safe_endgame_move(board, best_move_for_elsewhere, player):
                return best_move_for_elsewhere
        
        # 策略3: 如果没有好的移动，使用启发式选择
        print("使用启发式选择终局移动...")
        all_ai_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        if not all_ai_moves:
            return None
        
        # 选择最符合终局策略的移动
        return self._select_best_endgame_move_heuristic(board, all_ai_moves, player)
    
    def _find_path_to_region(self, board, start_piece, target_region, board_player, max_depth=6):
        """
        使用BFS找到从棋子到目标区域的最短路径
        """
        # 缓存键
        cache_key = (hash(start_piece), target_region, hash(str(board.cells)))
        if cache_key in self.endgame_cache:
            return self.endgame_cache[cache_key]
        
        queue = deque()
        visited = set()
        
        # (当前位置, 路径)
        queue.append((start_piece, [start_piece]))
        visited.add(start_piece)
        
        best_path = None
        best_length = float('inf')
        
        while queue:
            current_pos, path = queue.popleft()
            
            # 如果已经在目标区域
            if board.get_region(current_pos) == target_region:
                if len(path) < best_length:
                    best_length = len(path)
                    best_path = path
                continue
            
            # 限制搜索深度
            if len(path) > max_depth:
                continue
            
            # 生成所有可能的单步移动和单次跳跃
            for move in self._generate_simple_moves(board, current_pos, board_player, visited):
                if len(move) < 2:
                    continue
                    
                next_pos = move[-1]
                if next_pos not in visited:
                    new_path = path + [next_pos]
                    visited.add(next_pos)
                    queue.append((next_pos, new_path))
        
        # 缓存结果
        self.endgame_cache[cache_key] = best_path
        
        return best_path
    
    def _find_direct_move_to_region(self, board, piece, target_region, board_player):
        """查找直接进入目标区域的移动"""
        # 检查单步移动
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if (board.is_valid_cell(neighbor) and 
                board.is_empty(neighbor) and 
                board.get_region(neighbor) == target_region):
                return [piece, neighbor]
        
        # 检查单次跳跃
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if not board.is_valid_cell(neighbor) or board.is_empty(neighbor):
                continue
            
            jump_target = piece.neighbor(direction * 2)
            if (board.is_valid_cell(jump_target) and 
                board.is_empty(jump_target) and 
                board.get_region(jump_target) == target_region and
                ChineseCheckersMoves._is_valid_jump(board, piece, jump_target)):
                return [piece, jump_target]
        
        return None
    
    def _generate_simple_moves(self, board, piece, board_player, visited):
        """生成简单的移动（单步或单次跳跃）"""
        moves = []
        
        # 单步移动
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if (board.is_valid_cell(neighbor) and 
                board.is_empty(neighbor) and 
                neighbor not in visited):
                moves.append([piece, neighbor])
        
        # 单次跳跃
        for direction in range(6):
            neighbor = piece.neighbor(direction)
            if not board.is_valid_cell(neighbor) or board.is_empty(neighbor):
                continue
            
            # 尝试跳跃2步
            for dist in [2, 3, 4]:
                jump_target = piece + (CubeCoord.hex_directions[direction] * dist)
                if (board.is_valid_cell(jump_target) and 
                    board.is_empty(jump_target) and 
                    jump_target not in visited and
                    ChineseCheckersMoves._is_valid_jump(board, piece, jump_target)):
                    moves.append([piece, jump_target])
                    break
        
        return moves
    
    def _is_safe_endgame_move(self, board, move, player):
        """检查终局移动是否安全（不阻挡其他棋子）"""
        if len(move) < 2:
            return False
        
        start = move[0]
        end = move[-1]
        board_player = self.to_board_player(player)
        
        # 模拟移动
        test_board = ChineseCheckersMoves.apply_move(board, move)
        
        # 检查是否阻挡了其他AI棋子
        ai_pieces = test_board.get_player_pieces(board_player)
        for piece in ai_pieces:
            if piece == end:
                continue
            
            # 检查该棋子是否还能移动
            piece_moves = ChineseCheckersMoves.generate_moves_for_piece(test_board, piece)
            if not piece_moves:
                # 如果棋子被完全阻挡，检查是否是必要的
                if not self._is_piece_in_final_position(test_board, piece, board_player):
                    return False
        
        return True
    
    def _is_piece_in_final_position(self, board, piece, board_player):
        """检查棋子是否已经在最终位置（目标区域）"""
        target_region = board.player_target_regions[board_player]
        return board.get_region(piece) == target_region
    
    def _evaluate_endgame_path(self, path_move, start_piece, target_region):
        """评估终局路径的质量"""
        if len(path_move) < 2:
            return -float('inf')
        
        score = 0
        
        # 路径越短越好
        score -= len(path_move) * 20
        
        # 终点是否在目标区域
        end_piece = path_move[-1]
        # 这里我们假设path_move是一个坐标列表，实际需要根据实现调整
        
        return score
    
    def _select_best_endgame_move_heuristic(self, board, all_moves, player):
        """启发式选择终局移动"""
        if not all_moves:
            return None
        
        board_player = self.to_board_player(player)
        ai_target_region = 'tri0'  # AI的目标
        opponent_target_region = 'tri3'  # 对方坑位
        
        scored_moves = []
        
        for move in all_moves:
            if len(move) < 2:
                continue
                
            start = move[0]
            end = move[-1]
            start_region = board.get_region(start)
            end_region = board.get_region(end)
            
            score = 0
            
            # 1. 移动到AI目标区域（最高优先级）
            if end_region == ai_target_region:
                score += 1000
            
            # 2. 从对方坑位移动到AI目标区域
            elif start_region == opponent_target_region and end_region == ai_target_region:
                score += 800
            
            # 3. 移动到对方坑位
            elif end_region == opponent_target_region:
                score += 600
            
            # 4. 向AI目标区域前进
            elif self._is_moving_toward_target(start_region, end_region, ai_target_region):
                score += 200
            
            # 5. 跳跃奖励
            if len(move) > 2:
                score += 50 * (len(move) - 1)
            elif not start.is_neighbor(end):
                score += 30
            
            # 6. 避免后退
            if self._is_moving_away_from_target(start_region, end_region, ai_target_region):
                score -= 100
            
            scored_moves.append((score, move))
        
        if not scored_moves:
            return all_moves[0]
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return scored_moves[0][1]
    
    def _is_moving_toward_target(self, start_region, end_region, target_region):
        """检查是否向目标区域移动"""
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            target_idx = region_order.index(target_region)
            
            start_dist = abs(start_idx - target_idx)
            end_dist = abs(end_idx - target_idx)
            
            return end_dist < start_dist
        except ValueError:
            return False
    
    def _is_moving_away_from_target(self, start_region, end_region, target_region):
        """检查是否远离目标区域"""
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            target_idx = region_order.index(target_region)
            
            start_dist = abs(start_idx - target_idx)
            end_dist = abs(end_idx - target_idx)
            
            return end_dist > start_dist
        except ValueError:
            return False
    
    # 中局搜索方法保持不变
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
        
        return score
    
    def _alpha_beta_enhanced(self, board, depth, is_maximizing, alpha, beta, player):
        """Alpha-Beta搜索核心"""
        self.nodes_evaluated += 1
        
        if depth == 0:
            return self.evaluate_midgame(board, player)
        
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        board_player = self.to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self.evaluate_midgame(board, player)
        
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