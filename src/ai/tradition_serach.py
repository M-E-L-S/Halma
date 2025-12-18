import numpy as np
from typing import List, Tuple, Optional
import math
from collections import deque
import time
from src.core.board import ChineseCheckersBoard, CubeCoord
from src.core.moves import ChineseCheckersMoves

# 检查evaluator是否存在
try:
    from .evaluator import ChineseCheckersEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    print("警告: ChineseCheckersEvaluator未找到，使用简单评估器")

class Search:
    def __init__(self, player, depth=3):
        """
        初始化跳棋搜索AI
        参数:
            player: 玩家编号 (1 或 2)
            depth: 搜索深度
        """
        self.eva = None
        self.player = player
        self.depth = depth
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # 终局BFS搜索参数
        self.ENDGAME_THRESHOLD = 6       # 进入终局的棋子数阈值
        self.BFS_MAX_DEPTH = 15          # BFS最大搜索深度
        self.BFS_MAX_BRANCHES = 20      # BFS每层最大分支
        
        # 目标区域定义
        if player == 1:
            self.target_region_name = 'tri3'  # 玩家1的目标区域
        else:
            self.target_region_name = 'tri0'  # 玩家2的目标区域
    
    def get_opponent(self, player):
        """获取对手编号"""
        return -1 if player == 1 else 1
    
    def _to_board_player(self, player):
        """将1或2转换为1或-1供board使用"""
        return 1 if player == 1 else -1
    
    def _from_board_player(self, board_player):
        """将1或-1转换为1或2"""
        return 1 if board_player == 1 else -1
    
    def possible_moves(self, board, player):
        """
        生成所有可能的移动
        """
        board_player = self._to_board_player(player)
        return ChineseCheckersMoves.generate_all_moves(board, board_player)
    
    def find_immediate_win(self, board, player):
        """
        查找立即获胜的移动
        """
        moves = self.possible_moves(board, player)
        
        for move in moves:
            new_board = ChineseCheckersMoves.apply_move(board, move)
            if self.check_winner(new_board, player):
                return move
        
        return None
    
    def check_winner(self, board, player):
        """
        检查玩家是否获胜
        """
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        player_pieces = board.get_player_pieces(board_player)
        
        if len(player_pieces) != 10:
            return False
        
        for coord in player_pieces:
            if board.get_region(coord) != target_region:
                return False
        
        return True
    
    def is_in_endgame(self, board, player):
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
        """对移动进行排序"""
        if not moves:
            return []
        
        scored_moves = []
        
        for move in moves:
            score = self._evaluate_move(board, move, player)
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        max_moves = min(20, len(scored_moves))
        return [move for _, move in scored_moves[:max_moves]]
    
    def _evaluate_move(self, board, move, player):
        """评估单个移动的质量"""
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        end_region = board.get_region(end)
        
        if end_region == target_region:
            score += 1000
        
        start_region = board.get_region(start)
        start_dist = self._region_distance_to_target(start_region, player)
        end_dist = self._region_distance_to_target(end_region, player)
        
        if end_dist < start_dist:
            score += 500 * (start_dist - end_dist)
        
        if len(move) > 2:
            score += 100 * (len(move) - 1)
        elif not start.is_neighbor(end):
            score += 50
        
        empty_neighbors = 0
        for direction in range(6):
            neighbor = end.neighbor(direction)
            if board.is_valid_cell(neighbor) and board.is_empty(neighbor):
                empty_neighbors += 1
        
        score += empty_neighbors * 10
        
        center = self._find_center(board)
        end_distance_to_center = end.distance(center)
        if end_distance_to_center <= 2:
            score += 50 * (3 - end_distance_to_center)
        
        return score
    
    def _region_distance_to_target(self, region, player):
        """计算区域到目标区域的距离"""
        target_region = 'tri3' if player == 1 else 'tri0'
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        try:
            region_idx = region_order.index(region)
            target_idx = region_order.index(target_region)
            return abs(target_idx - region_idx)
        except ValueError:
            return 3
    
    def _find_center(self, board):
        """找到棋盘中心点"""
        hex_cells = [coord for coord, region in board.regions.items() 
                    if region == 'hex']
        if hex_cells:
            total_q = sum(c.q for c in hex_cells)
            total_r = sum(c.r for c in hex_cells)
            count = len(hex_cells)
            return CubeCoord(total_q // count, total_r // count)
        
        return next(iter(board.cells.keys()))
    
    def make_move(self, game_state, time_limit=3.0):
        """
        选择最佳移动 - 主接口
        参数:
            game_state: 游戏状态字典
            time_limit: 时间限制（秒）
        返回:
            最佳移动
        """
        print(f"AI 玩家{self.player} 开始思考...")
        start_time = time.time()
        
        board = game_state['board']
        current_player = game_state['current_player']
        
        # 初始化评估器
        if EVALUATOR_AVAILABLE:
            self.eva = ChineseCheckersEvaluator(board)
        else:
            # 使用简单评估器
            class SimpleEvaluator:
                def evaluate(self, board_cells, player):
                    return 0
            self.eva = SimpleEvaluator()
        
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
            print("使用优化终局搜索...")
            
            best_move = self._optimized_endgame_search(board, current_player, all_moves)
            if best_move:
                print(f"找到终局最优移动")
                return best_move
            else:
                print("终局搜索失败，回退到Alpha-Beta搜索")
        
        # 检查对手威胁
        opponent = self.get_opponent(current_player)
        opponent_win = self.find_immediate_win(board, opponent)
        if opponent_win:
            print("发现对手有立即获胜的威胁，尝试阻止")
            blocking_moves = self._find_blocking_moves(board, opponent_win, current_player)
            if blocking_moves:
                ordered = self.order_moves(board, blocking_moves, current_player)
                if ordered:
                    return ordered[0]
        
        # 使用Alpha-Beta搜索
        best_score = -float('inf')
        best_move = None
        
        ordered_moves = self.order_moves(board, all_moves, current_player)
        alpha = -float('inf')
        beta = float('inf')
        
        print(f"搜索深度: {self.depth}, 移动数: {len(ordered_moves)}")
        
        for i, move in enumerate(ordered_moves[:15]):
            # 时间检查
            if time.time() - start_time > time_limit - 0.5:
                print(f"时间限制到达，返回当前最佳移动")
                break
            
            new_board = ChineseCheckersMoves.apply_move(board, move)
            
            score = self._alpha_beta(
                new_board, 
                self.depth - 1, 
                False,
                alpha, 
                beta, 
                opponent
            )
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        elapsed = time.time() - start_time
        print(f"搜索完成: {elapsed:.2f}秒")
        print(f"评估节点数: {self.nodes_evaluated}, 剪枝次数: {self.pruning_count}")
        print(f"最佳评估值: {best_score}")
        
        return best_move if best_move else (ordered_moves[0] if ordered_moves else None)
    
       # ==================== 优化终局搜索 - 专门处理最后一个棋子 ====================
    
    def _optimized_endgame_search(self, board, player, all_moves):
        """
        优化终局搜索：专注让外面的棋子进入目标区域
        特别优化只剩1-2个棋子在外面的情况
        """
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        print(f"优化终局搜索：玩家{player}，目标区域{target_region}")
        
        # 1. 找出所有不在目标区域的棋子
        player_pieces = board.get_player_pieces(board_player)
        pieces_outside = []
        pieces_inside = []
        
        for piece in player_pieces:
            if board.get_region(piece) != target_region:
                pieces_outside.append(piece)
            else:
                pieces_inside.append(piece)
        
        outside_count = len(pieces_outside)
        print(f"外面棋子: {outside_count}个，里面棋子: {len(pieces_inside)}个")
        
        # 2. 如果所有棋子都在目标区域内，选择最少破坏性的移动
        if outside_count == 0:
            print("所有棋子都在目标区域内，选择稳定移动")
            return self._select_stable_move(board, player, all_moves)
        
        # 3. 如果是只剩1-2个棋子在外面，使用专门的搜索策略
        if outside_count <= 2:
            print(f"只剩{outside_count}个棋子在外面，使用专门搜索策略...")
            return self._final_pieces_search(board, player, pieces_outside, pieces_inside, all_moves)
        
        # 4. 一般情况：优先考虑让外面的棋子进入目标区域
        print("优先搜索外面棋子进入目标区域的移动...")
        
        # 4.1 首先检查是否有外面棋子直接进入目标区域的移动
        outside_entering_moves = []
        for move in all_moves:
            start = move[0]
            end = move[-1]
            
            if board.get_region(start) != target_region:
                if board.get_region(end) == target_region:
                    outside_entering_moves.append(move)
        
        if outside_entering_moves:
            print(f"找到{len(outside_entering_moves)}个直接进入目标区域的移动")
            outside_entering_moves.sort(key=lambda m: len(m))
            return outside_entering_moves[0]
        
        # 4.2 为外面的棋子搜索最短路径
        print("为外面的棋子搜索最短路径...")
        
        best_moves_for_outside = []
        for piece in pieces_outside:
            # 使用优化的BFS搜索
            path = self._optimized_bfs_for_last_piece(board, piece, board_player, target_region)
            
            if path and len(path) >= 2:
                # 查找对应的移动
                move = self._find_best_move_for_path(board, piece, path, board_player)
                if move:
                    score = self._evaluate_outside_move(board, move, player, piece, target_region)
                    best_moves_for_outside.append((score, move, piece))
        
        if best_moves_for_outside:
            best_moves_for_outside.sort(reverse=True, key=lambda x: x[0])
            best_score, best_move, best_piece = best_moves_for_outside[0]
            print(f"选择外面棋子{best_piece}的移动，得分: {best_score:.1f}")
            return best_move
        
        # 5. 如果没有好的外部棋子移动，选择前进方向最好的移动
        print("选择前进方向最好的移动...")
        return self._select_best_forward_move(board, player, all_moves)
    

    def _select_best_forward_move(self, board, player, all_moves):
        """选择前进方向最好的移动"""
        best_move = None
        best_score = -float('inf')
        
        for move in all_moves[:20]:  # 只检查前20个移动
            score = self._evaluate_move(board, move, player)  # 使用已有的评估函数
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else (all_moves[0] if all_moves else None)
    def _final_pieces_search(self, board, player, pieces_outside, pieces_inside, all_moves):
        """
        专门处理最后1-2个棋子的搜索策略
        """
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        print(f"专门处理最后{len(pieces_outside)}个棋子的搜索...")
        
        # 如果是只剩1个棋子在外面，全力让它进入
        if len(pieces_outside) == 1:
            last_piece = pieces_outside[0]
            print(f"最后一个棋子: {last_piece}")
            
            # 1. 检查是否有直接进入目标区域的移动
            direct_moves = []
            for move in all_moves:
                if move[0] == last_piece and board.get_region(move[-1]) == target_region:
                    direct_moves.append(move)
            
            if direct_moves:
                print("找到直接进入目标区域的移动")
                direct_moves.sort(key=lambda m: len(m))
                return direct_moves[0]
            
            # 2. 使用A*算法搜索最短路径
            print("使用A*搜索最短路径...")
            best_path = self._a_star_search_for_last_piece(board, last_piece, board_player, target_region)
            
            if best_path and len(best_path) >= 2:
                # 找到对应的移动
                move = self._find_exact_move_for_path(board, last_piece, best_path, board_player)
                if move:
                    print(f"找到A*路径: {len(best_path)-1}步")
                    return move
            
            # 3. 如果A*找不到，使用改进的BFS
            print("使用改进BFS搜索...")
            bfs_path = self._enhanced_bfs_for_last_piece(board, last_piece, board_player, target_region)
            
            if bfs_path and len(bfs_path) >= 2:
                move = self._find_exact_move_for_path(board, last_piece, bfs_path, board_player)
                if move:
                    print(f"找到BFS路径: {len(bfs_path)-1}步")
                    return move
        
        # 对于2个棋子的情况，优先让它们轮流前进
        elif len(pieces_outside) == 2:
            print("两个棋子在外面，优化轮流前进...")
            return self._two_pieces_strategy(board, player, pieces_outside, pieces_inside, all_moves)
        
        # 如果上面的方法都失败，使用启发式
        print("使用启发式选择...")
        return self._select_best_forward_move(board, player, all_moves)
    
    def _a_star_search_for_last_piece(self, board, start_piece, board_player, target_region):
        """
        A*算法搜索最短路径（最后一个棋子专用）
        """
        from queue import PriorityQueue
        
        # 优先级队列：(f_score, g_score, position, path)
        pq = PriorityQueue()
        g_score = {start_piece: 0}  # 从起点到当前点的实际代价
        f_score = {start_piece: self._heuristic(start_piece, target_region)}  # 估计总代价
        
        pq.put((f_score[start_piece], 0, start_piece, [start_piece]))
        
        visited = set()
        
        while not pq.empty():
            current_f, current_g, current_pos, path = pq.get()
            
            # 如果已经在目标区域，返回路径
            if board.get_region(current_pos) == target_region:
                return path
            
            if current_pos in visited:
                continue
            visited.add(current_pos)
            
            # 获取所有可能的下一步
            next_positions = self._get_all_possible_moves_for_position(board, current_pos, board_player)
            
            for next_pos in next_positions:
                if next_pos in visited:
                    continue
                
                # 计算新的g_score（每步代价为1）
                tentative_g = current_g + 1
                
                if next_pos not in g_score or tentative_g < g_score[next_pos]:
                    # 找到更好的路径
                    new_path = path + [next_pos]
                    g_score[next_pos] = tentative_g
                    h = self._heuristic(next_pos, target_region)
                    f_score[next_pos] = tentative_g + h
                    
                    pq.put((f_score[next_pos], tentative_g, next_pos, new_path))
        
        return None
    
    def _heuristic(self, position, target_region):
        """
        A*启发式函数：估计到目标区域的距离
        """
        # 简单启发式：基于区域距离
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        # 这里需要知道position所在的区域，但position是CubeCoord
        # 我们需要从board获取区域信息，但这里没有board参数
        # 修改：返回一个保守估计
        return 0  # 暂时返回0，让A*退化为BFS
    
    def _get_all_possible_moves_for_position(self, board, position, board_player):
        """
        获取某个位置的所有可能移动目标位置
        """
        positions = []
        
        # 获取所有移动
        moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        for move in moves:
            if move[0] == position:
                positions.append(move[-1])
        
        return positions
    
    def _enhanced_bfs_for_last_piece(self, board, start_piece, board_player, target_region):
        """
        增强的BFS搜索，专门为最后一个棋子设计
        """
        queue = deque()
        visited = {}
        parent = {}
        
        queue.append((start_piece, 0))  # (位置, 步数)
        visited[start_piece] = 0
        parent[start_piece] = None
        
        found_target = None
        min_steps = float('inf')
        
        max_steps = 20  # 最大搜索步数
        
        while queue:
            current_pos, steps = queue.popleft()
            
            # 如果已经到达目标区域
            if board.get_region(current_pos) == target_region:
                if steps < min_steps:
                    min_steps = steps
                    found_target = current_pos
                continue
            
            if steps >= max_steps:
                continue
            
            # 获取所有可能的下一步
            next_positions = self._get_smart_moves_for_last_piece(board, current_pos, board_player, target_region, visited)
            
            for next_pos in next_positions:
                if next_pos not in visited or steps + 1 < visited[next_pos]:
                    visited[next_pos] = steps + 1
                    parent[next_pos] = current_pos
                    queue.append((next_pos, steps + 1))
        
        # 重构路径
        if found_target:
            path = []
            current = found_target
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        
        return None
    
    def _get_smart_moves_for_last_piece(self, board, position, board_player, target_region, visited):
        """
        为最后一个棋子获取智能移动选项
        优先考虑向目标区域前进的移动
        """
        positions = []
        
        # 获取所有移动
        moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        for move in moves:
            if move[0] == position:
                end_pos = move[-1]
                
                # 跳过已访问的位置（除非能更快到达）
                if end_pos in visited and len(move) >= visited.get(end_pos, 0):
                    continue
                
                positions.append(end_pos)
        
        # 按启发式排序：越靠近目标区域越好
        positions.sort(key=lambda pos: self._distance_to_target_region(pos, board, target_region))
        
        return positions[:8]  # 只考虑前8个最好的选择
    
    def _distance_to_target_region(self, position, board, target_region):
        """
        计算位置到目标区域的距离
        """
        current_region = board.get_region(position)
        
        if current_region == target_region:
            return 0
        
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        try:
            current_idx = region_order.index(current_region)
            target_idx = region_order.index(target_region)
            return abs(current_idx - target_idx)
        except ValueError:
            return 10
    
    def _find_exact_move_for_path(self, board, start_piece, path, board_player):
        """
        为路径查找确切的移动序列
        尝试找到从起点到终点的完整移动
        """
        if len(path) < 2:
            return None
        
        # 查找单步移动
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # 查找从start到end的移动
            moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
            for move in moves:
                if move[0] == start and move[-1] == end:
                    # 如果是路径的最后一步，返回这个移动
                    if end == path[-1]:
                        return move
                    # 否则继续查找
                    break
        
        # 如果找不到完整路径，返回第一步
        if len(path) >= 2:
            first_end = path[1]
            moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
            for move in moves:
                if move[0] == start_piece and move[-1] == first_end:
                    return move
        
        return None
    
    def _two_pieces_strategy(self, board, player, pieces_outside, pieces_inside, all_moves):
        """
        两个棋子在外面的策略
        """
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        # 策略1：优先让更接近目标区域的棋子前进
        piece1, piece2 = pieces_outside[0], pieces_outside[1]
        
        # 计算每个棋子到目标区域的距离
        dist1 = self._distance_to_target_region(piece1, board, target_region)
        dist2 = self._distance_to_target_region(piece2, board, target_region)
        
        # 选择距离更近的棋子优先移动
        if dist1 <= dist2:
            primary_piece = piece1
            secondary_piece = piece2
        else:
            primary_piece = piece2
            secondary_piece = piece1
        
        print(f"优先移动棋子{primary_piece}（距离{min(dist1, dist2)}）")
        
        # 查找主要棋子的移动
        primary_moves = []
        for move in all_moves:
            if move[0] == primary_piece:
                primary_moves.append(move)
        
        if primary_moves:
            # 优先选择能进入目标区域的移动
            for move in primary_moves:
                if board.get_region(move[-1]) == target_region:
                    return move
            
            # 其次选择前进距离最大的移动
            primary_moves.sort(
                key=lambda m: self._calculate_advancement(m[0], m[-1], board, player),
                reverse=True
            )
            return primary_moves[0]
        
        # 如果主要棋子没有移动，尝试次要棋子
        secondary_moves = []
        for move in all_moves:
            if move[0] == secondary_piece:
                secondary_moves.append(move)
        
        if secondary_moves:
            secondary_moves.sort(
                key=lambda m: self._calculate_advancement(m[0], m[-1], board, player),
                reverse=True
            )
            return secondary_moves[0]
        
        # 如果都没有，返回第一个移动
        return all_moves[0] if all_moves else None
    
    def _calculate_advancement(self, start, end, board, player):
        """计算前进程度"""
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        start_region = board.get_region(start)
        end_region = board.get_region(end)
        
        if end_region == target_region:
            return 1000  # 进入目标区域
        
        region_order = ['tri0', 'tri5', 'hex', 'tri4', 'tri3', 'tri2', 'tri1']
        
        try:
            start_idx = region_order.index(start_region)
            end_idx = region_order.index(end_region)
            target_idx = region_order.index(target_region)
            
            # 计算前进距离
            start_dist = abs(start_idx - target_idx)
            end_dist = abs(end_idx - target_idx)
            
            return start_dist - end_dist  # 正值表示前进
        except ValueError:
            return 0
    
    def _optimized_bfs_for_last_piece(self, board, start_piece, board_player, target_region):
        """
        优化BFS搜索，考虑跳跃移动
        """
        # 使用队列存储（位置，路径，步数）
        queue = deque()
        visited = {}
        
        queue.append((start_piece, [start_piece], 0))
        visited[start_piece] = 0
        
        best_path = None
        min_steps = float('inf')
        
        while queue:
            current_pos, path, steps = queue.popleft()
            
            # 如果到达目标区域，记录最佳路径
            if board.get_region(current_pos) == target_region:
                if steps < min_steps:
                    min_steps = steps
                    best_path = path
                continue
            
            if steps >= 15:  # 限制搜索深度
                continue
            
            # 获取所有可能的移动
            moves = self._get_moves_for_last_piece_bfs(board, current_pos, board_player)
            
            for move in moves:
                next_pos = move[-1]
                next_steps = steps + 1
                
                # 如果没访问过或者找到更短路径
                if next_pos not in visited or next_steps < visited[next_pos]:
                    visited[next_pos] = next_steps
                    new_path = path + [next_pos]
                    queue.append((next_pos, new_path, next_steps))
        
        return best_path
    
    def _get_moves_for_last_piece_bfs(self, board, position, board_player):
        """
        获取BFS搜索中的移动选项
        """
        moves = []
        
        # 单步移动
        for direction in range(6):
            neighbor = position.neighbor(direction)
            if board.is_valid_cell(neighbor) and board.is_empty(neighbor):
                moves.append([position, neighbor])
        
        # 跳跃移动
        all_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        for move in all_moves:
            if move[0] == position:
                moves.append(move)
        
        return moves
    
    def _find_best_move_for_path(self, board, start_piece, path, board_player):
        """
        为路径查找最佳移动
        """
        if not path or len(path) < 2:
            return None
        
        # 尝试找到从起点到终点的移动
        end_pos = path[-1]
        
        moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        for move in moves:
            if move[0] == start_piece and move[-1] == end_pos:
                return move
        
        # 如果找不到，尝试第一步
        if len(path) >= 2:
            first_end = path[1]
            for move in moves:
                if move[0] == start_piece and move[-1] == first_end:
                    return move
        
        return None
    
    def _evaluate_outside_move(self, board, move, player, piece, target_region):
        """
        评估外面棋子移动的质量
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        # 验证起点是否正确
        if start != piece:
            return -10000
        
        # 是否进入目标区域
        if board.get_region(end) == target_region:
            score += 10000
        
        # 前进距离
        start_region = board.get_region(start)
        end_region = board.get_region(end)
        
        if start_region != target_region and end_region == target_region:
            # 从外面进入目标区域，额外奖励
            score += 5000
        
        # 移动效率
        steps = len(move) - 1
        if steps == 1:
            score += 500  # 单步效率高
        elif steps <= 3:
            score += 300
        else:
            score -= (steps - 3) * 100
        
        return score
    # ==================== Alpha-Beta搜索核心 ====================
    
    def _alpha_beta(self, board, depth, is_maximizing, alpha, beta, player):
        """Alpha-Beta剪枝核心算法"""
        self.nodes_evaluated += 1
        
        if depth == 0:
            return self.evaluate_board(board)
        
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        board_player = self._to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            return self.evaluate_board(board)
        
        ordered_moves = self.order_moves(board, valid_moves, player)
        
        if is_maximizing:
            max_eval = -float('inf')
            
            for move in ordered_moves[:12]:
                new_board = ChineseCheckersMoves.apply_move(board, move)
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
                new_board = ChineseCheckersMoves.apply_move(board, move)
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
            return 0
        
        board_player = self._to_board_player(self.player)
        
        if hasattr(self.eva, 'evaluate'):
            if EVALUATOR_AVAILABLE:
                return self.eva.evaluate(board.cells.copy(), board_player)
            else:
                # 使用简单评估
                return self._simple_evaluate(board, self.player)
        
        return 0
    
    def _simple_evaluate(self, board, player):
        """简单棋盘评估"""
        score = 0
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        
        # 计算在目标区域的棋子数
        player_pieces = board.get_player_pieces(board_player)
        in_target = 0
        for piece in player_pieces:
            if board.get_region(piece) == target_region:
                in_target += 1
                score += 200
        
        # 前进程度
        for piece in player_pieces:
            region = board.get_region(piece)
            dist = self._region_distance_to_target(region, player)
            score -= dist * 50
        
        # 中心控制
        center = CubeCoord(0, 0, 0)
        for piece in player_pieces:
            dist = piece.distance(center)
            if dist <= 3:
                score += 20 * (4 - dist)
        
        return score