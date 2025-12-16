import numpy as np
from typing import List, Tuple, Optional
from src.core.board import ChineseCheckersBoard, CubeCoord
from src.core.moves import ChineseCheckersMoves

class Search:
    def __init__(self, player, depth=3):
        self.player = player
        self.depth = depth
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
    def get_opponent(self, player):
        """获取对手编号"""
        return 2 if player == 1 else 1
    
    def possible_moves(self, board, player):
        """获取当前玩家所有可能的移动"""
        move_player = 1 if player == 1 else -1
        return ChineseCheckersMoves.generate_all_moves(board, move_player)
    
    def find_immediate_win(self, board, player):
        """
        查找立即获胜的移动
        
        参数:
            board: 棋盘对象
            player: 当前玩家 (1 或 2)
            
        返回:
            立即获胜的移动，如果没有返回None
        """
        # 转换玩家编号用于移动生成
        move_player = 1 if player == 1 else -1
        moves = self.possible_moves(board, player)
        
        for move in moves:
            # 模拟移动
            test_board = board.copy()
            start = move[0]
            end = move[-1]
            
            # 执行移动
            piece = test_board.get_piece(start)
            test_board.set_piece(start, 0)
            test_board.set_piece(end, piece)
            
            # 检查是否获胜（使用1和2编号）
            if self.check_winner(test_board, player):
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
        board_player = 1 if player == 1 else -1
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
        board_player = 1 if player == 1 else -1
        
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
            score += 200 * (start_dist - end_dist)
        
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
        
        # 获取棋盘状态
        board = game_state['board']
        current_player = game_state['current_player']
        
        # 重置计数
        self.nodes_evaluated = 0
        self.pruning_count = 0
        print(f"AI玩家{self.player}正在思考...")
        
        # 获取所有合法移动
        all_moves = game_state['valid_moves']
        
        if not all_moves:
            return None
            
        if len(all_moves) == 1:
            return all_moves[0]
        
        # 先检查是否有立即获胜的移动
        win_move = self.find_immediate_win(board, current_player)
        if win_move:
            return win_move
        
        # 检查对手是否有立即获胜的威胁
        opponent = self.get_opponent(current_player)
        opponent_win = self.find_immediate_win(board, opponent)
        if opponent_win:
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
        
        for i, move in enumerate(ordered_moves[:15]):  # 限制搜索分支
            # 模拟移动
            test_board = board.copy()
            start = move[0]
            end = move[-1]
            piece = test_board.get_piece(start)
            test_board.set_piece(start, 0)
            test_board.set_piece(end, piece)
            
            # 递归搜索
            if current_player == self.player:
                score = self._alpha_beta(test_board, self.depth - 1, False, 
                                        alpha, beta, opponent)
            else:
                score = self._alpha_beta(test_board, self.depth - 1, True,
                                        alpha, beta, current_player)
            
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
        
        return best_move if best_move else ordered_moves[0]
    
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
            return self.evaluate_board(board,playerid)
        
        # 检查游戏是否结束
        if self.check_winner(board, self.player):
            return 10000 + depth * 100
        elif self.check_winner(board, self.get_opponent(self.player)):
            return -10000 - depth * 100
        
        # 生成当前玩家的所有合法移动
        move_player = 1 if player == 1 else -1
        valid_moves = ChineseCheckersMoves.generate_all_moves(board, move_player)
        
        if not valid_moves:
            # 如果没有合法移动，返回评估值
            return self.evaluate_board(board , playerid)
        
        # 对移动排序
        ordered_moves = self.order_moves(board, valid_moves, player)
        
        if is_maximizing:
            max_eval = -float('inf')
            
            for move in ordered_moves[:12]:  # 限制分支
                # 模拟移动
                test_board = board.copy()
                start = move[0]
                end = move[-1]
                piece = test_board.get_piece(start)
                test_board.set_piece(start, 0)
                test_board.set_piece(end, piece)
                
                # 对手回合
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta(
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
            
            for move in ordered_moves[:12]:
                test_board = board.copy()
                start = move[0]
                end = move[-1]
                piece = test_board.get_piece(start)
                test_board.set_piece(start, 0)
                test_board.set_piece(end, piece)
                
                # 对手回合
                next_player = self.get_opponent(player)
                eval_score = self._alpha_beta(
                    test_board, depth - 1, True, alpha, beta, next_player
                )
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    self.pruning_count += 1
                    break
            
            return min_eval
    
    def evaluate_board(self, board, player=None):
        pass
    
    def _count_pieces_in_target(self, board, player):
        """计算在目标区域的棋子数"""
        board_player = 1 if player == 1 else -1
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
        board_player = 1 if player == 1 else -1
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
        board_player = 1 if self.player == 1 else -1
        for piece in board.get_player_pieces(board_player):
            distance = piece.distance(center)
            if distance <= 3:
                ai_score += (4 - distance)
        
        # 对手棋子
        opponent = self.get_opponent(self.player)
        board_opponent = 1 if opponent == 1 else -1
        for piece in board.get_player_pieces(board_opponent):
            distance = piece.distance(center)
            if distance <= 3:
                opponent_score += (4 - distance)
        
        return ai_score - opponent_score
    
    def _calculate_connectivity(self, board):
        """计算棋子连接度差异"""
        def calculate_for_player(player_pieces):
            if not player_pieces:
                return 0
            
            pieces_set = set(player_pieces)
            total_connections = 0
            
            for piece in pieces_set:
                # 计算相邻的己方棋子
                for direction in range(6):
                    neighbor = piece.neighbor(direction)
                    if neighbor in pieces_set:
                        total_connections += 1
            
            return total_connections / 2  # 每条连接被计算了两次
        
        # AI玩家
        board_player = 1 if self.player == 1 else -1
        ai_connections = calculate_for_player(board.get_player_pieces(board_player))
        
        # 对手
        opponent = self.get_opponent(self.player)
        board_opponent = 1 if opponent == 1 else -1
        opponent_connections = calculate_for_player(board.get_player_pieces(board_opponent))
        
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
            for piece in board.get_player_pieces(1 if player == 1 else -1):
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
                    for piece in board.get_player_pieces(1 if player == 1 else -1):
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
            all_moves = ChineseCheckersMoves.generate_all_moves(board, piece_owner)
            for move in all_moves:
                if move[0] == start and move[-1] == target:
                    moves.append(move)
        
        return moves


# 测试函数
def test_search_ai():
    """测试跳棋搜索AI"""
    print("测试跳棋搜索AI...")
    
    from src.core.game_state import ChineseCheckersGame
    
    # 创建游戏
    game = ChineseCheckersGame()
    
    # 创建搜索AI（玩家1）
    search_ai = Search(player=1, depth=2)
    
    # 获取游戏状态
    state = game.get_state()
    
    # 测试AI搜索
    print("\nAI玩家1正在搜索最佳移动...")
    best_move = search_ai.make_move(state)
    
    if best_move:
        print(f"AI建议的移动: {best_move}")
        
        # 执行移动
        success, message = game.make_move(best_move)
        print(f"执行结果: {success}, 消息: {message}")
        
        # 显示更新后的状态
        game.print_status()
        
        # 测试对手AI（玩家2）
        print("\nAI玩家2正在搜索最佳移动...")
        search_ai2 = Search(player=2, depth=2)
        state2 = game.get_state()
        best_move2 = search_ai2.make_move(state2)
        
        if best_move2:
            print(f"AI玩家2建议的移动: {best_move2}")
    else:
        print("AI没有找到合法移动")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_search_ai()