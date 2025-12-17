import numpy as np
from typing import List, Tuple, Optional
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
        
        self.max_branch_factor = 20  # 每层最大搜索分支数
        self.early_move_factor = 12  # 早期搜索分支数（深度较大时减少）
    
    def get_opponent(self, player):
        return 2 if player == 1 else 1
    
    def possible_moves(self, board, player):
        move_player = 1 if player == 1 else -1
        return ChineseCheckersMoves.generate_all_moves(board, move_player)
    
    def find_immediate_win(self, board, player):
        moves = self.possible_moves(board, player)
        for move in moves:
            test_board = board.copy()
            test_board = ChineseCheckersMoves.apply_move(test_board, move)
            if self.check_winner(test_board, player):
                return move
        return None
    
    def check_winner(self, board, player):
        board_player = 1 if player == 1 else 2
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        if len(player_pieces) != 10:
            return False
        
        for coord in player_pieces:
            if board.get_region(coord) != target_region:
                return False
        
        return True
    
    def greedy_order_moves(self, board, moves, player, depth):
        """
        贪心算法排序移动：对每个移动进行快速评估，选择得分最高的
        
        参数:
            board: 棋盘对象
            moves: 移动列表
            player: 当前玩家
            depth: 当前深度
            
        返回:
            排序后的移动列表
        """
        if not moves:
            return []
        
        scored_moves = []
        
        for move in moves:
            # 快速评估移动质量（比完整评估快）
            score = self._quick_evaluate_move(board, move, player, depth)
            scored_moves.append((score, move))
        
        # 按分数降序排序（最大化玩家要高分数，最小化玩家要低分数）
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        # 根据深度动态调整分支数
        if depth > 2:  # 深层搜索减少分支
            max_moves = min(self.early_move_factor, len(scored_moves))
        else:
            max_moves = min(self.max_branch_factor, len(scored_moves))
        
        return [move for _, move in scored_moves[:max_moves]]
    
    def _quick_evaluate_move(self, board, move, player, depth):
    
        score = 0
        start = move[0]
        end = move[-1]
        
        # 转换玩家编号
        board_player = 1 if player == 1 else 2
        
        # 1. 目标区域奖励（最重要）
        target_region = board.player_target_regions[board_player]
        end_region = board.get_region(end)
        
        if end_region == target_region:
            score += 500  # 快速评估中权重稍低
        
        # 2. 前进距离
        start_region = board.get_region(start)
        if start_region != end_region:
            # 简单判断是否向前进
            region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
            target_region_name = 'tri3' if player == 1 else 'tri0'
            
            try:
                start_idx = region_order.index(start_region)
                end_idx = region_order.index(end_region)
                target_idx = region_order.index(target_region_name)
                
                # 计算移动后的距离变化
                start_dist = abs(start_idx - target_idx)
                end_dist = abs(end_idx - target_idx)
                
                if end_dist < start_dist:
                    score += 100 * (start_dist - end_dist)
            except:
                pass
        
        # 3. 跳跃奖励（但深层搜索时减少奖励，避免过度跳跃）
        if len(move) > 2:  # 连续跳跃
            jump_bonus = 30 if depth > 1 else 50
            score += jump_bonus * (len(move) - 1)
        elif not start.is_neighbor(end):  # 单次跳跃
            jump_bonus = 20 if depth > 1 else 30
            score += jump_bonus
        
        # 4. 防御性考虑：如果对手有威胁，优先防守
        opponent = self.get_opponent(player)
        if depth <= 1:  # 浅层时考虑防守
            opponent_moves = self.possible_moves(board, opponent)
            for opp_move in opponent_moves[:5]:  # 只检查前几个对手移动
                opp_end = opp_move[-1]
                if opp_end == end:  # 可能阻挡对手
                    score += 80
        
        return score
    
    def make_move(self, game_state):
        """选择最佳移动 - 主接口"""

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
            return win_move
        
        # 使用贪心算法增强的Alpha-Beta搜索
        best_score = -float('inf')
        best_move = None
        
        # 对根节点移动进行贪心排序
        root_moves = self.greedy_order_moves(board, all_moves, current_player, 0)
        
        alpha = -float('inf')
        beta = float('inf')
        
        for i, move in enumerate(root_moves):
            test_board = ChineseCheckersMoves.apply_move(board, move)
            
            # 递归搜索
            if current_player == self.player:
                score = self._alpha_beta_enhanced(test_board, self.depth - 1, 
                                                False, alpha, beta, 
                                                self.get_opponent(current_player))
            else:
                score = self._alpha_beta_enhanced(test_board, self.depth - 1,
                                                True, alpha, beta, current_player)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            # Alpha-Beta剪枝
            if current_player == self.player:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            
            if beta <= alpha:
                self.pruning_count += 1
                break
        return best_move if best_move else root_moves[0]
    
    def _alpha_beta_enhanced(self, board, depth, is_maximizing, alpha, beta, player):
        """
        增强的Alpha-Beta剪枝算法（带贪心排序）
        
        参数:
            board: 棋盘状态
            depth: 剩余深度
            is_maximizing: 是否为最大化玩家
            alpha: α值
            beta: β值
            player: 当前玩家
            
        返回:
            评估分数
        """
        self.nodes_evaluated += 1
        
        # 终止条件
        if depth == 0:
            return self.evaluate_board(board, player)
        
        # 检查游戏是否结束
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        # 生成当前玩家的所有合法移动
        board_player = 1 if player == 1 else 2
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self.evaluate_board(board, player)
        
        # 使用贪心算法对移动排序
        ordered_moves = self.greedy_order_moves(board, valid_moves, player, depth)
        
        if is_maximizing:
            max_eval = -float('inf')
            
            for move in ordered_moves:
                # 模拟移动
                test_board = ChineseCheckersMoves.apply_move(board, move)
                
                # 对手回合
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta_enhanced(
                    test_board, depth - 1, False, alpha, beta, next_player
                )
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # 剪枝
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            
            for move in ordered_moves:
                test_board = ChineseCheckersMoves.apply_move(board, move)
                
                # 对手回合
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta_enhanced(
                    test_board, depth - 1, True, alpha, beta, next_player
                )
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # 剪枝
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            return min_eval
    
    def evaluate_board(self, board, player):
       return self.eva.evaluate(board.cells.copy(), player)
    
    def _count_pieces_in_target(self, board, board_player):
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        count = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                count += 1
        return count
    
    def _calculate_progress(self, board, player):
        board_player = 1 if player == 1 else 2
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        if not player_pieces:
            return 0
        
        total_progress = 0
        for piece in player_pieces:
            piece_region = board.get_region(piece)
            if piece_region == target_region:
                total_progress += 10
            elif piece_region == 'hex':
                total_progress += 5
            else:
                region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
                try:
                    region_idx = region_order.index(piece_region)
                    target_idx = region_order.index(target_region)
                    distance = abs(target_idx - region_idx)
                    total_progress += max(0, 5 - distance)
                except:
                    total_progress += 2
        
        return total_progress / len(player_pieces)
    
    def _calculate_center_control(self, board):
        center = self._find_center(board)
        ai_score = 0
        opponent_score = 0
        
        board_player = 1 if self.player == 1 else 2
        for piece in board.get_player_pieces(board_player):
            distance = piece.distance(center)
            if distance <= 3:
                ai_score += (4 - distance)
        
        opponent = self.get_opponent(self.player)
        board_opponent = 1 if opponent == 1 else 2
        for piece in board.get_player_pieces(board_opponent):
            distance = piece.distance(center)
            if distance <= 3:
                opponent_score += (4 - distance)
        
        return ai_score - opponent_score
    
    def _calculate_connectivity(self, board):
        def calculate_for_player(player_pieces):
            if not player_pieces:
                return 0
            
            pieces_set = set(player_pieces)
            total_connections = 0
            
            for piece in pieces_set:
                for direction in range(6):
                    neighbor = piece.neighbor(direction)
                    if neighbor in pieces_set:
                        total_connections += 1
            
            return total_connections / 2
        
        board_player = 1 if self.player == 1 else 2
        ai_connections = calculate_for_player(board.get_player_pieces(board_player))
        
        opponent = self.get_opponent(self.player)
        board_opponent = 1 if opponent == 1 else 2
        opponent_connections = calculate_for_player(board.get_player_pieces(board_opponent))
        
        return ai_connections - opponent_connections
    
    def _calculate_safety(self, board, player):
        """计算棋子安全性（避免孤立棋子）"""
        board_player = 1 if player == 1 else 2
        player_pieces = board.get_player_pieces(board_player)
        
        safe_score = 0
        for piece in player_pieces:
            # 检查棋子是否有己方邻居
            has_friend = False
            for direction in range(6):
                neighbor = piece.neighbor(direction)
                if neighbor in player_pieces:
                    has_friend = True
                    break
            
            if has_friend:
                safe_score += 2
            else:
                safe_score -= 1  # 孤立棋子惩罚
        
        return safe_score
    
    def _find_center(self, board):
        hex_cells = [coord for coord, region in board.regions.items() 
                    if region == 'hex']
        if hex_cells:
            total_q = sum(c.q for c in hex_cells)
            total_r = sum(c.r for c in hex_cells)
            count = len(hex_cells)
            return CubeCoord(total_q // count, total_r // count)
        
        return next(iter(board.cells.keys()))
    



