import numpy as np
from typing import List, Tuple, Optional
import math
from collections import deque
from src.core.board import ChineseCheckersBoard, CubeCoord
from src.core.moves import ChineseCheckersMoves
from .evaluator import ChineseCheckersEvaluator

class Search:
    def __init__(self, player, depth=3):
        """
        初始化跳棋搜索AI
        
        参数:
            player: 玩家编号 (1 或 2)
            depth: 搜索深度
        """
        self.eva = None
        self.player = player  # 保持1或2格式
        self.depth = depth
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # 终局BFS搜索参数
        self.ENDGAME_THRESHOLD = 7       # 进入终局的棋子数阈值
        self.BFS_MAX_DEPTH = 8           # BFS最大搜索深度
        self.BFS_MAX_BRANCHES = 8        # BFS每层最大分支
        
        # 目标区域定义
        if player == 1:
            self.target_region_name = 'tri3'  # 玩家1的目标区域
        else:
            self.target_region_name = 'tri0'  # 玩家2的目标区域
    
    def get_opponent(self, player):
        """获取对手编号"""
        return 2 if player == 1 else 1  # 修正：返回2或1，不是-1
    
    def _to_board_player(self, player):
        """将1或2转换为1或-1供board使用"""
        return 1 if player == 1 else -1
    
    def _from_board_player(self, board_player):
        """将1或-1转换为1或2"""
        return 1 if board_player == 1 else 2
    
    def possible_moves(self, board, player):
        """
        生成所有可能的移动 - 使用现有的移动生成器
        
        参数:
            board: 跳棋棋盘对象
            player: 当前玩家 (1 或 2)
            
        返回:
            所有合法移动列表
        """
        # 转换为board格式
        board_player = self._to_board_player(player)
        return ChineseCheckersMoves.generate_all_moves(board, board_player)
    
    def find_immediate_win(self, board, player):
        """
        查找立即获胜的移动
        
        参数:
            board: 棋盘对象
            player: 当前玩家 (1 或 2)
            
        返回:
            立即获胜的移动，如果没有返回None
        """
        moves = self.possible_moves(board, player)
        
        for move in moves:
            # 使用正确的移动应用方法
            new_board = ChineseCheckersMoves.apply_move(board, move)
            
            # 检查是否获胜
            if self.check_winner(new_board, player):
                return move
        
        return None
    
    def check_winner(self, board, player):
        """
        检查玩家是否获胜
        
        参数:
            board: 棋盘对象
            player: 玩家编号 (1 或 2)
            
        返回:
            bool: 是否获胜
        """
        # 转换玩家编号用于获取目标区域
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        # 获取玩家棋子
        player_pieces = board.get_player_pieces(board_player)
        
        # 所有棋子都必须在目标区域
        if len(player_pieces) != 10:
            return False
        
        for coord in player_pieces:
            if board.get_region(coord) != target_region:
                return False
        
        return True
    
    def is_in_endgame(self, board, player):
        """检查是否进入终局阶段"""
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        in_target = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                in_target += 1
        
        return in_target >= self.ENDGAME_THRESHOLD
    
    def get_pieces_in_target(self, board, player):
        """获取在目标区域的棋子数"""
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        in_target = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                in_target += 1
        
        return in_target
    
    def order_moves(self, board, moves, player):
        """
        对移动进行排序，优先搜索有希望的移动
        
        参数:
            board: 棋盘对象
            moves: 移动列表
            player: 当前玩家 (1 或 2)
            
        返回:
            排序后的移动列表
        """
        if not moves:
            return []
        
        scored_moves = []
        
        for move in moves:
            score = self._evaluate_move(board, move, player)
            scored_moves.append((score, move))
        
        # 按分数降序排序
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        # 只返回前N个最好的移动以提高搜索效率
        max_moves = min(20, len(scored_moves))
        return [move for _, move in scored_moves[:max_moves]]
    
    def _evaluate_move(self, board, move, player):
        """
        评估单个移动的质量
        
        参数:
            board: 棋盘对象
            move: 移动
            player: 当前玩家 (1 或 2)
            
        返回:
            移动评分
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        # 转换玩家编号用于获取目标区域
        board_player = self._to_board_player(player)
        
        # 1. 目标区域奖励
        target_region = board.player_target_regions[board_player]
        end_region = board.get_region(end)
        
        if end_region == target_region:
            score += 1000
        
        # 2. 前进距离奖励
        start_region = board.get_region(start)
        start_dist = self._region_distance_to_target(start_region, player)
        end_dist = self._region_distance_to_target(end_region, player)
        
        if end_dist < start_dist:  # 向目标前进
            score += 500 * (start_dist - end_dist)
        
        # 3. 跳跃奖励
        if len(move) > 2:  # 连续跳跃
            score += 100 * (len(move) - 1)
        elif not start.is_neighbor(end):  # 单次跳跃
            score += 50
        
        # 4. 移动后的灵活性
        empty_neighbors = 0
        for direction in range(6):
            neighbor = end.neighbor(direction)
            if board.is_valid_cell(neighbor) and board.is_empty(neighbor):
                empty_neighbors += 1
        
        score += empty_neighbors * 10
        
        # 5. 中心控制奖励
        center = self._find_center(board)
        end_distance_to_center = end.distance(center)
        if end_distance_to_center <= 2:
            score += 50 * (3 - end_distance_to_center)
        
        return score
    
    def _region_distance_to_target(self, region, player):
        """
        计算区域到目标区域的距离
        
        参数:
            region: 当前区域
            player: 当前玩家 (1 或 2)
            
        返回:
            距离值
        """
        # 玩家1目标：tri3，玩家2目标：tri0
        target_region = 'tri3' if player == 1 else 'tri0'
        
        # 区域顺序（按顺时针）
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        try:
            region_idx = region_order.index(region)
            target_idx = region_order.index(target_region)
            return abs(target_idx - region_idx)
        except ValueError:
            return 3  # 默认距离
    
    def _find_center(self, board):
        """找到棋盘中心点"""
        # 查找中心区域中的点
        hex_cells = [coord for coord, region in board.regions.items() 
                    if region == 'hex']
        if hex_cells:
            # 返回大致中心
            total_q = sum(c.q for c in hex_cells)
            total_r = sum(c.r for c in hex_cells)
            count = len(hex_cells)
            return CubeCoord(total_q // count, total_r // count)
        
        # 如果找不到，返回第一个坐标
        return next(iter(board.cells.keys()))
    
    def make_move(self, game_state):
        """
        选择最佳移动 - 主接口
        
        参数:
            game_state: 游戏状态字典
            
        返回:
            最佳移动
        """
        print(f"AI 玩家{self.player} 开始思考...")
        
        # 获取棋盘状态
        board = game_state['board']
        current_player = game_state['current_player']  # 来自game_state
        
        # 初始化评估器
        self.eva = ChineseCheckersEvaluator(board)  # 传递board对象
        
        # 重置计数
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # 获取所有合法移动
        all_moves = game_state['valid_moves']
        
        if not all_moves:
            print("没有合法移动")
            return None
            
        if len(all_moves) == 1:
            print("只有一个合法移动，直接返回")
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
            print("使用BFS搜索最优填坑路径...")
            
            # 使用BFS搜索终局最优路径
            best_move = self._endgame_bfs_search(board, current_player, all_moves)
            if best_move:
                print(f"BFS找到终局最优移动")
                return best_move
            else:
                print("BFS搜索失败，回退到Alpha-Beta搜索")
        
        # 检查对手是否有立即获胜的威胁
        opponent = self.get_opponent(current_player)
        opponent_win = self.find_immediate_win(board, opponent)
        if opponent_win:
            print("发现对手有立即获胜的威胁，尝试阻止")
            # 尝试阻止对手
            blocking_moves = self._find_blocking_moves(board, opponent_win, current_player)
            if blocking_moves:
                ordered = self.order_moves(board, blocking_moves, current_player)
                if ordered:
                    return ordered[0]
        
        # 使用Alpha-Beta搜索
        best_score = -float('inf')
        best_move = None
        
        # 对移动排序
        ordered_moves = self.order_moves(board, all_moves, current_player)
        
        alpha = -float('inf')
        beta = float('inf')
        
        print(f"搜索深度: {self.depth}, 移动数: {len(ordered_moves)}")
        
        for i, move in enumerate(ordered_moves[:15]):  # 限制搜索分支
            # 使用正确的移动应用方法
            new_board = ChineseCheckersMoves.apply_move(board, move)
            
            # 递归搜索
            score = self._alpha_beta(
                new_board, 
                self.depth - 1, 
                False,  # 对手回合
                alpha, 
                beta, 
                opponent  # 下一个玩家是对手
            )
            
            if score > best_score:
                best_score = score
                best_move = move
            
            # Alpha-Beta剪枝
            alpha = max(alpha, score)
            
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        print(f"评估节点数: {self.nodes_evaluated}, 剪枝次数: {self.pruning_count}")
        print(f"最佳评估值: {best_score}")
        
        return best_move if best_move else ordered_moves[0]
    
    # ==================== 终局BFS搜索 ====================
    
    def _endgame_bfs_search(self, board, player, all_moves):
        """
        BFS广度优先搜索终局最优路径
        专注于找到最短的进入目标区域的路径
        """
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
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
            
            if shortest_path and len(shortest_path) >= 2:
                # 将路径转换为移动
                path_move = self._convert_path_to_move(board, piece, shortest_path, board_player)
                if path_move:
                    # 评估路径的质量
                    path_score = self._evaluate_endgame_path(board, path_move, player)
                    best_move_for_pieces.append((path_score, path_move))
                    print(f"  找到路径: {len(shortest_path)-1}步, 得分: {path_score:.1f}")
        
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
        # 使用BFS队列：(当前位置, 路径)
        queue = deque()
        visited = set()
        
        # 初始化队列
        queue.append((start_piece, [start_piece]))
        visited.add(start_piece)
        
        best_path = None
        best_length = float('inf')
        
        depth = 0
        while queue and depth < self.BFS_MAX_DEPTH:
            level_size = len(queue)
            
            for _ in range(level_size):
                current_pos, path = queue.popleft()
                
                # 如果当前位置已经在目标区域，更新最佳路径
                if board.get_region(current_pos) == target_region:
                    path_length = len(path) - 1  # 减去起点
                    if path_length < best_length:
                        best_length = path_length
                        best_path = path
                    continue
                
                # 生成当前棋子的所有可能移动
                next_moves = self._get_moves_for_position(board, current_pos, board_player, visited)
                
                for move in next_moves[:self.BFS_MAX_BRANCHES]:  # 限制分支
                    next_pos = move[-1]
                    
                    if next_pos not in visited:
                        new_path = path + [next_pos]
                        visited.add(next_pos)
                        queue.append((next_pos, new_path))
            
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
                # 检查是否是有效的跳跃
                if len(jump) >= 2:  # 至少包含起点和终点
                    moves.append(jump)
        
        # 3. 按启发式排序移动
        moves.sort(key=lambda m: self._heuristic_distance(m[-1], board, board_player))
        
        return moves
    
    def _heuristic_distance(self, position, board, board_player):
        """
        计算位置到目标区域的启发式距离
        值越小表示越接近目标
        """
        target_region = board.player_target_regions[board_player]
        
        # 如果已经在目标区域，距离为0
        if board.get_region(position) == target_region:
            return 0
        
        # 基于区域距离的启发式
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        current_region = board.get_region(position)
        
        try:
            current_idx = region_order.index(current_region)
            target_idx = region_order.index(target_region)
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
        
        # 如果没有直接移动，尝试构建多步移动
        # 这里简化处理：取路径的前两步
        if len(path) >= 2:
            first_step = path[1]
            # 查找从起点到第一步的移动
            for move in all_moves:
                if move[0] == start and move[-1] == first_step:
                    return move
        
        # 如果找不到，返回空
        return None
    
    def _evaluate_endgame_path(self, board, move, player):
        """
        评估终局路径的质量
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        # 1. 是否进入目标区域（最高奖励）
        if board.get_region(end) == target_region:
            score += 500
        
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
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            target_idx = region_order.index(target_region)
            
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
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
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
        
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        # 1. 是否进入目标区域
        if board.get_region(end) == target_region:
            score += 500
        
        # 2. 距离目标区域的变化
        start_region = board.get_region(start)
        end_region = board.get_region(end)
        
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            target_idx = region_order.index(target_region)
            
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
    
    def _alpha_beta(self, board, depth, is_maximizing, alpha, beta, player):
        """
        Alpha-Beta剪枝核心算法
        
        参数:
            board: 棋盘状态
            depth: 剩余深度
            is_maximizing: 是否为最大化玩家
            alpha: α值
            beta: β值
            player: 当前玩家 (1 或 2)
            
        返回:
            评估分数
        """
        self.nodes_evaluated += 1
        
        # 终止条件
        if depth == 0:
            return self.evaluate_board(board)
        
        # 检查游戏是否结束
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        # 生成当前玩家的所有合法移动
        board_player = self._to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            # 如果没有合法移动，返回评估值
            return self.evaluate_board(board)
        
        # 对移动排序
        ordered_moves = self.order_moves(board, valid_moves, player)
        
        if is_maximizing:
            max_eval = -float('inf')
            
            for move in ordered_moves[:12]:  # 限制分支
                # 使用正确的移动应用方法
                new_board = ChineseCheckersMoves.apply_move(board, move)
                
                # 对手回合
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta(
                    new_board, depth - 1, False, alpha, beta, next_player
                )
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            
            for move in ordered_moves[:12]:
                # 使用正确的移动应用方法
                new_board = ChineseCheckersMoves.apply_move(board, move)
                
                # 对手回合
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta(
                    new_board, depth - 1, True, alpha, beta, next_player
                )
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            return min_eval
    
    def evaluate_board(self, board):
        """评估棋盘状态"""
        if self.eva is None:
            # 如果评估器未初始化，创建一个临时评估器
            self.eva = ChineseCheckersEvaluator(board)
        
        # 注意：evaluator可能期望1/-1格式
        board_player = self._to_board_player(self.player)
        return self.eva.evaluate(board.cells.copy(), board_player)
    
    def _count_pieces_in_target(self, board, player):
        """计算在目标区域的棋子数"""
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        count = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                count += 1
        
        return count
    
    def _calculate_progress(self, board, player):
        """计算玩家前进进度"""
        # player参数是1或2格式
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        if not player_pieces:
            return 0
        
        total_progress = 0
        for piece in player_pieces:
            piece_region = board.get_region(piece)
            if piece_region == target_region:
                total_progress += 10  # 在目标区域
            elif piece_region == 'hex':
                total_progress += 5   # 在中心区域
            else:
                # 根据区域计算进度
                region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
                try:
                    region_idx = region_order.index(piece_region)
                    target_idx = region_order.index(target_region)
                    distance = abs(target_idx - region_idx)
                    total_progress += max(0, 5 - distance)  # 距离越近进度越高
                except:
                    total_progress += 2
        
        return total_progress / len(player_pieces)
    
    def _calculate_center_control(self, board):
        """计算中心控制度"""
        center = self._find_center(board)
        ai_score = 0
        opponent_score = 0
        
        # AI玩家棋子
        board_ai = self._to_board_player(self.player)
        for piece in board.get_player_pieces(board_ai):
            distance = piece.distance(center)
            if distance <= 3:
                ai_score += (4 - distance)
        
        # 对手棋子
        opponent = self.get_opponent(self.player)
        board_opponent = self._to_board_player(opponent)
        for piece in board.get_player_pieces(board_opponent):
            distance = piece.distance(center)
            if distance <= 3:
                opponent_score += (4 - distance)
        
        return ai_score - opponent_score
    
    def _calculate_connectivity(self, board):
        """计算棋子连接度差异"""
        def calculate_for_player(board_player):
            pieces = board.get_player_pieces(board_player)
            if not pieces:
                return 0
            
            pieces_set = set(pieces)
            total_connections = 0
            
            for piece in pieces_set:
                # 计算相邻的己方棋子
                for direction in range(6):
                    neighbor = piece.neighbor(direction)
                    if neighbor in pieces_set:
                        total_connections += 1
            
            return total_connections / 2  # 每条连接被计算了两次
        
        # AI玩家
        board_ai = self._to_board_player(self.player)
        ai_connections = calculate_for_player(board_ai)
        
        # 对手
        opponent = self.get_opponent(self.player)
        board_opponent = self._to_board_player(opponent)
        opponent_connections = calculate_for_player(board_opponent)
        
        return ai_connections - opponent_connections
    
    def _find_blocking_moves(self, board, opponent_win_move, player):
        """
        查找阻止对手获胜的移动
        
        参数:
            board: 棋盘对象
            opponent_win_move: 对手的获胜移动
            player: 当前玩家 (1 或 2)
            
        返回:
            可能的阻挡移动列表
        """
        blocking_moves = []
        target = opponent_win_move[-1]  # 对手的目标位置
        
        # 尝试占领对手的目标位置
        if board.is_empty(target):
            # 查找能到达该位置的棋子
            board_player = self._to_board_player(player)
            for piece in board.get_player_pieces(board_player):
                # 检查是否能移动到目标位置
                moves = self._find_path_to_target(board, piece, target)
                if moves:
                    blocking_moves.extend(moves)
        
        # 尝试阻挡对手的路径
        if len(opponent_win_move) > 1:
            # 尝试在对手路径上放置棋子
            for i in range(1, len(opponent_win_move)):
                blocking_point = opponent_win_move[i]
                if board.is_empty(blocking_point):
                    # 查找能到达该位置的棋子
                    board_player = self._to_board_player(player)
                    for piece in board.get_player_pieces(board_player):
                        moves = self._find_path_to_target(board, piece, blocking_point)
                        if moves:
                            blocking_moves.extend(moves)
        
        return blocking_moves
    
    def _find_path_to_target(self, board, start, target):
        """
        查找从起点到目标的路径
        
        参数:
            board: 棋盘对象
            start: 起始位置
            target: 目标位置
            
        返回:
            可能的移动列表
        """
        moves = []
        
        # 检查单步移动
        if start.is_neighbor(target) and board.is_empty(target):
            moves.append([start, target])
        
        # 检查跳跃移动
        # 获取起始位置的棋子所有者
        piece_owner = board.get_piece(start)
        if piece_owner != 0:
            # 将棋子所有者转换为1或2格式
            player_num = 1 if piece_owner == 1 else 2
            all_moves = self.possible_moves(board, player_num)
            for move in all_moves:
                if move[0] == start and move[-1] == target:
                    moves.append(move)
        
        return moves