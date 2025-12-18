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
        
        # 搜索参数 - 控制确定性的节点数
        self.MAX_BRANCH_FACTOR = 8       # 每层最大分支数（确定节点数的关键）
        self.EARLY_BRANCH_FACTOR = 12    # 前几层稍多的分支数
        self.DFS_DEPTH_LIMIT = 5         # DFS搜索深度限制
        
        # 终局参数
        self.ENDGAME_THRESHOLD = 6
        self.endgame_cache = {}
    
    def to_board_player(self, player):
        """将玩家编号(1/2)转换为棋盘表示(1/-1)"""
        return 1 if player == 1 else -1
    
    def get_opponent(self, player):
        """获取对手编号"""
        return 2 if player == 1 else 1
    
    def get_target_region_name(self, player):
        """获取目标区域名称"""
        return 'tri3' if player == 1 else 'tri0'
    
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
        
        if is_endgame:
            print(f"进入终局阶段！使用终局搜索...")
            best_move = self._endgame_search(board, current_player, all_moves)
            if best_move:
                return best_move
        
        # 使用Alpha-Beta搜索
        print("使用Alpha-Beta搜索...")
        best_move = self._deterministic_alpha_beta_search(board, current_player, all_moves)
        
        return best_move
    
    def _deterministic_alpha_beta_search(self, board, player, all_moves):
        """
        确定性的Alpha-Beta搜索
        保证搜索的节点数是确定的（通过固定分支因子）
        """
        # 第1步：对根节点的移动进行贪心排序
        sorted_root_moves = self._greedy_sort_moves(board, all_moves, player, 0)
        
        # 限制根节点的分支数，保证确定性
        max_root_branches = min(self.EARLY_BRANCH_FACTOR, len(sorted_root_moves))
        root_moves = sorted_root_moves[:max_root_branches]
        
        print(f"根节点搜索分支数: {len(root_moves)}/{len(all_moves)}")
        
        # Alpha-Beta搜索参数
        alpha = -float('inf')
        beta = float('inf')
        best_score = -float('inf')
        best_move = root_moves[0] if root_moves else None
        
        # 第2步：对每个候选移动进行Alpha-Beta搜索
        for i, move in enumerate(root_moves):
            # 应用移动
            new_board = ChineseCheckersMoves.apply_move(board, move)
            
            # 递归搜索（Min层）
            if player == self.player:
                score = self._alpha_beta_min(new_board, self.depth-1, alpha, beta, 
                                           self.get_opponent(player))
            else:
                score = self._alpha_beta_max(new_board, self.depth-1, alpha, beta, 
                                           self.get_opponent(player))
            
            print(f"移动 {i+1}/{len(root_moves)}: {move[0]}→{move[-1]}, 得分: {score:.1f}")
            
            # 更新最佳移动
            if score > best_score:
                best_score = score
                best_move = move
            
            # Alpha-Beta剪枝
            if player == self.player:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            
            # 剪枝检查
            if beta <= alpha:
                self.pruning_count += 1
                print(f"在第{i+1}个移动处剪枝")
                break
        
        # 输出统计信息
        total_possible_nodes = self._calculate_possible_nodes(self.depth)
        efficiency = self.nodes_evaluated / total_possible_nodes if total_possible_nodes > 0 else 0
    
        return best_move
    
    def _alpha_beta_max(self, board, depth, alpha, beta, player):
        """Max层搜索"""
        self.nodes_evaluated += 1
        
        # 终止条件
        if depth == 0:
            return self._evaluate_board(board, player)
        
        # 检查游戏结束
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        # 生成合法移动
        board_player = self.to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self._evaluate_board(board, player)
        
        # 贪心排序并限制分支数
        sorted_moves = self._greedy_sort_moves(board, valid_moves, player, depth)
        max_branches = self._get_branch_factor(depth)
        moves_to_search = sorted_moves[:max_branches]
        
        # 搜索最佳值
        max_eval = -float('inf')
        
        for move in moves_to_search:
            new_board = ChineseCheckersMoves.apply_move(board, move)
            next_player = self.get_opponent(player)
            eval_score = self._alpha_beta_min(new_board, depth-1, alpha, beta, next_player)
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            # Alpha-Beta剪枝
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        return max_eval
    
    def _alpha_beta_min(self, board, depth, alpha, beta, player):
        """Min层搜索"""
        self.nodes_evaluated += 1
        
        # 终止条件
        if depth == 0:
            return self._evaluate_board(board, player)
        
        # 检查游戏结束
        if self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        elif self.check_winner(board, self.player):
            return 10000 + depth * 100
        
        # 生成合法移动
        board_player = self.to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self._evaluate_board(board, player)
        
        # 贪心排序并限制分支数
        sorted_moves = self._greedy_sort_moves(board, valid_moves, player, depth)
        max_branches = self._get_branch_factor(depth)
        moves_to_search = sorted_moves[:max_branches]
        
        # 搜索最小值
        min_eval = float('inf')
        
        for move in moves_to_search:
            new_board = ChineseCheckersMoves.apply_move(board, move)
            next_player = self.get_opponent(player)
            eval_score = self._alpha_beta_max(new_board, depth-1, alpha, beta, next_player)
            
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            
            # Alpha-Beta剪枝
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        return min_eval
    
    
    def _greedy_sort_moves(self, board, moves, player, depth):
        """
        贪心排序移动 - 快速评估移动质量并排序
        保证确定性的排序结果
        """
        if not moves:
            return []
        
        scored_moves = []
        for move in moves:
            score = self._quick_move_evaluation(board, move, player, depth)
            scored_moves.append((score, move))
        
        # 按分数降序排序
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        # 返回排序后的移动列表
        return [move for _, move in scored_moves]
    
    def _quick_move_evaluation(self, board, move, player, depth):
        """
        快速评估移动质量
        使用简单的启发式规则
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self.to_board_player(player)
        target_region = board.player_target_regions[board_player]
        end_region = board.get_region(end)
        start_region = board.get_region(start)
        
        # 1. 目标区域奖励（最高优先级）
        if end_region == target_region:
            score += 2000
        
        # 2. 前进方向奖励
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        target_name = self.get_target_region_name(player)
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            target_idx = region_order.index(target_name)
            
            # 计算距离改进
            start_dist = abs(start_idx - target_idx)
            end_dist = abs(end_idx - target_idx)
            
            if end_dist < start_dist:  # 向目标前进
                score += 300 * (start_dist - end_dist)
            elif end_dist > start_dist:  # 后退
                score -= 500
            
            # 根据深度调整权重
            if depth > 0:
                score *= 0.9  # 深层的移动评估稍微降低权重
        except ValueError:
            pass
        
        # 3. 跳跃奖励
        jump_count = len(move) - 1
        if jump_count > 1:
            score += 80 * jump_count  # 跳跃奖励
        elif not start.is_neighbor(end):
            score += 50  # 单次跳跃
        
        # 4. 中心控制
        center = CubeCoord(0, 0, 0)
        end_distance = end.distance(center)
        if end_distance <= 2:
            score += 30 * (3 - end_distance)
        
        # 5. 棋子连接性（鼓励棋子聚集）
        # 计算移动后有多少邻居是己方棋子
        friendly_neighbors = 0
        for direction in range(6):
            neighbor = end.neighbor(direction)
            if (board.is_valid_cell(neighbor) and 
                board.get_piece(neighbor) == board_player):
                friendly_neighbors += 1
        
        score += friendly_neighbors * 15
        
        return score
    
    
    def _evaluate_board(self, board, player):
        """评估棋盘状态"""
        if self.eva is None:
            self.eva = ChineseCheckersEvaluator(board)
        
        board_player = self.to_board_player(player)
        return self.eva.evaluate(board.cells.copy(), board_player)
    
    
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
        if depth >= self.depth - 1:  # 接近根节点的层
            return self.EARLY_BRANCH_FACTOR
        else:
            return self.MAX_BRANCH_FACTOR
    
    def _calculate_possible_nodes(self, depth):
        """
        计算理论最大节点数
        用于评估搜索效率
        """
        if depth <= 0:
            return 1
        
        # 根节点使用early分支因子
        total_nodes = self.EARLY_BRANCH_FACTOR
        
        # 计算后续层的节点数
        for d in range(1, depth):
            if d <= 1:  # 前两层使用early分支因子
                total_nodes *= self.EARLY_BRANCH_FACTOR
            else:  # 后续层使用max分支因子
                total_nodes *= self.MAX_BRANCH_FACTOR
        
        return total_nodes
    
    def _endgame_search(self, board, player, all_moves):
        """终局搜索（简化版）"""
        print("使用简化终局搜索...")
        
        # 简单策略：优先选择进入目标区域的移动
        board_player = self.to_board_player(player)
        target_region = self.get_target_region_name(player)
        
        best_move = None
        best_score = -float('inf')
        
        for move in all_moves[:min(10, len(all_moves))]:  # 只检查前10个移动
            end = move[-1]
            
            # 如果移动到目标区域
            if board.get_region(end) == target_region:
                # 进一步评估这个移动
                score = self._quick_move_evaluation(board, move, player, 0)
                if score > best_score:
                    best_score = score
                    best_move = move
        
        if best_move:
            print(f"终局搜索找到移动: {best_move[0]}→{best_move[-1]}, 得分: {best_score:.1f}")
            return best_move
        
        # 如果没有找到进入目标区域的移动，返回第一个移动
        return all_moves[0] if all_moves else None