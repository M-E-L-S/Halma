import time
from src.core.board import ChineseCheckersBoard, CubeCoord
from src.core.moves import ChineseCheckersMoves

# 检查evaluator是否存在
try:
    from src.ai.evaluator import ChineseCheckersEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    print("警告: ChineseCheckersEvaluator未找到，使用简单评估器")

class Search:
    def __init__(self, player, depth=3):
        """
        初始化跳棋搜索AI
        参数:
            player: 玩家编号 (1 或 -1)
            depth: 搜索深度
        """
        self.eva = None
        self.player = player
        self.depth = depth
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # 终局BFS搜索参数
        self.ENDGAME_THRESHOLD = 8       # 进入终局的棋子数阈值
        self.min_distance_threshold = 2  # 双方棋子最小距离阈值（新增）
        self.MAX_BFS_DEPTH = 10          # 终局BFS最大搜索深度

        self.BFS_PATH = []               # 终局BFS路径存储
        self.BFS_FOUND = False           # 终局BFS是否找到路径标志
        self.ENDGAME_FLAG = False        # 终局阶段标志

        self.target_region_name = 'tri0' if player == -1 else 'tri3'
        self.opponent_target_for_ai = 'tri0' if player == -1 else 'tri3'
    
    def get_opponent(self, player):
        """获取对手编号（保持1/-1映射）"""
        return -1 if player == 1 else 1
    
    def _to_board_player(self, player):
        return 1 if player == 1 else -1
    
    def possible_moves(self, board, player):
        """生成所有可能的移动"""
        board_player = self._to_board_player(player)
        return ChineseCheckersMoves.generate_all_moves(board, board_player)
    
    def find_immediate_win(self, board, player):
        """查找立即获胜的移动"""
        moves = self.possible_moves(board, player)
        
        for move in moves:
            new_board = ChineseCheckersMoves.apply_move(board, move)
            if self.check_winner(new_board, player):
                return move
        
        return None
    
    def check_winner(self, board, player):
        """检查玩家是否获胜"""
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
        """
        修复后的终局判断逻辑（保留你的核心逻辑，仅修复错误）
        条件1: 双方棋子的最小距离大于2
        条件2: AI方有棋子进入对方坑位（tri3）
        """
        # 修复1：正确转换为board内部的玩家编号（1/-1），而非直接用self.player
        board_player = self._to_board_player(self.player)  # AI的内部编号（比如玩家2对应-1）
        opponent_board_player = - board_player              # 对手的内部编号（比如1）
        
        # 获取双方棋子
        ai_pieces = board.get_player_pieces(board_player)
        opponent_pieces = board.get_player_pieces(opponent_board_player)
        
        if not ai_pieces or not opponent_pieces:
            return False
        
        # 条件1: 检查最小距离是否大于2（保留你的逻辑，移除冗余判断）
        min_distance = float('inf')
        for ai_piece in ai_pieces:
            for opp_piece in opponent_pieces:
                distance = ai_piece.distance(opp_piece)
                if distance < min_distance:
                    min_distance = distance
                    if min_distance <= self.min_distance_threshold:
                        # 有棋子距离≤2，不满足终局条件
                        return False
        
        # 条件2: AI方是否有棋子进入对方坑位
        ai_pieces_in_opponent_target = 0
        for piece in ai_pieces:
            if board.get_region(piece) == self.opponent_target_for_ai:
                ai_pieces_in_opponent_target += 1
        
        return ai_pieces_in_opponent_target > 6
    
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
        """对移动进行排序（增加安全校验）"""
        if not moves:
            return []
        
        scored_moves = []
        
        for move in moves:
            try:
                score = self._evaluate_move(board, move, player)
                # 确保score是数值类型
                score = score if isinstance(score, (int, float)) else 0
            except Exception:
                score = 0
            scored_moves.append((score, move))
        
        # 安全排序：只比较分数
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        max_moves = min(50, len(scored_moves))
        return [move for _, move in scored_moves[:max_moves]]
    
    def _evaluate_move(self, board, move, player):
        """
        评估单个移动的质量（核心修复：区分外部进目标区和内部移动）
        """
        score = 0
        start = move[0]
        end = move[-1]
        
        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]
        start_in_target = (board.get_region(start) == target_region)
        end_in_target = (board.get_region(end) == target_region)

        # 1. 核心修复：差异化奖励
        if end_in_target:
            if not start_in_target:
                # 外部棋子进入目标区：超级高额奖励（优先级别最高）
                score += 10000
            else:
                # 目标区内移动：极低奖励（避免循环）
                score += 10
        else:
            # 2. 外部棋子前进奖励
            start_region = board.get_region(start)
            start_dist = self._region_distance_to_target(start_region, player)
            end_dist = self._region_distance_to_target(board.get_region(end), player)
            
            if end_dist < start_dist:
                score += 500 * (start_dist - end_dist)
        
        # 3. 跳跃奖励（鼓励高效移动）
        if len(move) > 2:
            score += 100 * (len(move) - 1)
        elif not start.is_neighbor(end):
            score += 50
        
        # 4. 移动后的灵活性（避免死局）
        empty_neighbors = 0
        for direction in range(6):
            neighbor = end.neighbor(direction)
            if board.is_valid_cell(neighbor) and board.is_empty(neighbor):
                empty_neighbors += 1
        
        score += empty_neighbors * 10
        
        # 5. 中心控制奖励（非终局阶段有效）
        center = self._find_center(board)
        end_distance_to_center = end.distance(center)
        if end_distance_to_center <= 2 and not start_in_target:
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
    
    def make_move(self, game_state, time_limit=60.0):
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
        if not self.ENDGAME_FLAG:
            self.ENDGAME_FLAG = self.is_in_endgame(board, current_player)
        
        if self.ENDGAME_FLAG:
            print("使用优化终局BFS搜索...")

            if not self.BFS_PATH:
                best_move = self._optimized_endgame_bfs(board, current_player, all_moves)
                if best_move:
                    print(f"找到终局最优移动")
                    return best_move
                else:
                    print("终局搜索失败，回退到Alpha-Beta搜索")
            else:
                print("终局路径已存在，继续使用BFS路径")
                self.BFS_PATH.pop(0)
                return self.BFS_PATH[0]

        # 检查对手威胁
        opponent = self.get_opponent(current_player)
        
        # 使用Alpha-Beta搜索
        best_score = -float('inf')
        best_move = None
        
        ordered_moves = self.order_moves(board, all_moves, current_player)
        alpha = -float('inf')
        beta = float('inf')
        
        print(f"搜索深度: {self.depth}, 移动数: {len(ordered_moves)}")
        
        for i, move in enumerate(ordered_moves):
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

    def _optimized_endgame_bfs(self, board, player, all_moves):
        from collections import deque

        board_player = self._to_board_player(player)
        target_region = board.player_target_regions[board_player]

        # 序列化棋盘用于判重
        def serialize(b):
            items = []
            for coord in sorted(b.cells.keys(), key=lambda c: (getattr(c, 'q', 0), getattr(c, 'r', 0),
                                                               getattr(c, 's', 0) if hasattr(c, 's') else 0)):
                owner = b.get_piece(coord) if hasattr(b, 'get_piece') else (b.cells.get(coord, 0))
                q = getattr(coord, 'q', 0)
                r = getattr(coord, 'r', 0)
                s = getattr(coord, 's', None)
                if s is None:
                    items.append((q, r, owner))
                else:
                    items.append((q, r, s, owner))
            return tuple(items)

        # 过滤规则：已在目标区的棋子不允许移出目标区；允许目标区内部移动或外部棋子任意移动
        def move_allowed(m, b):
            start = m[0]
            end = m[-1]
            start_region = b.get_region(start)
            end_region = b.get_region(end)
            if start_region == target_region and end_region != target_region:
                return False
            if start_region == target_region and end_region == target_region:
                return True
            if start_region != target_region:
                return end.distance(CubeCoord(q=6,r=-3,s=-3)) < start.distance(CubeCoord(q=6,r=-3,s=-3))
            return False

        q = deque()
        start_ser = serialize(board)
        q.append((board, [], 0))
        visited = set([start_ser])

        while q:
            cur_board, path, depth = q.popleft()
            if depth >= self.MAX_BFS_DEPTH:
                break

            valid_moves = ChineseCheckersMoves.generate_all_moves(cur_board, board_player)
            if not valid_moves:
                continue

            filtered = [m for m in valid_moves if move_allowed(m, cur_board)]
            if not filtered:
                continue

            for m in filtered:
                new_board = ChineseCheckersMoves.apply_move(cur_board, m)
                ser = serialize(new_board)
                if ser in visited:
                    continue
                visited.add(ser)

                new_path = path + [m]

                # 检查是否胜利（对外部 player）
                if self.check_winner(new_board, player):
                    self.BFS_FOUND = True
                    self.BFS_PATH = new_path
                    return self.BFS_PATH[0] if self.BFS_PATH else None

                q.append((new_board, new_path, depth + 1))

        self.BFS_FOUND = False
        self.BFS_PATH = []
        return None

    def _alpha_beta(self, board, depth, is_maximizing, alpha, beta, player):
        """Alpha-Beta剪枝核心算法"""
        self.nodes_evaluated += 1
        
        if depth == 0:
            return self.eva.evaluate(board.cells.copy(), player)
            #return self.eva.NNUE_evaluate(board)
        
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        board_player = self._to_board_player(player)
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, board_player)
        
        if not valid_moves:
            # 如果没有合法移动，返回评估值
            return self.eva.evaluate(board.cells.copy(), player)
        
        ordered_moves = self.order_moves(board, valid_moves, player)
        
        if is_maximizing:
            max_eval = -float('inf')
            
            for move in ordered_moves:
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
            
            for move in ordered_moves:
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