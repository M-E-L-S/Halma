# moves.py
<<<<<<< HEAD
from src.core.board import *
=======
from board import *
>>>>>>> 4df9e2bc9aeedfcc2694ab47d689d8b22fcbedb5
class ChineseCheckersMoves:
    """中国跳棋移动生成器"""

    @staticmethod
    def generate_all_moves(board, player):
        """生成玩家所有合法移动"""
        all_moves = []
        player_pieces = board.get_player_pieces(player)

        for start_coord in player_pieces:
            piece_moves = ChineseCheckersMoves.generate_moves_for_piece(board, start_coord)
            all_moves.extend(piece_moves)

        return all_moves

    @staticmethod
    def generate_moves_for_piece(board, start_coord):
        """生成单个棋子的所有可能移动"""
        moves = []

        # 1. 单步移动
        single_moves = ChineseCheckersMoves._generate_single_moves(board, start_coord)
        moves.extend(single_moves)

        # 2. 跳跃移动
        jump_moves = ChineseCheckersMoves._generate_jump_moves(board, start_coord, [start_coord])
        moves.extend(jump_moves)

        return moves

    @staticmethod
    def _generate_single_moves(board, coord):
        """生成单步移动"""
        moves = []

        for direction in range(6):
            neighbor = coord.neighbor(direction)
            if board.is_valid_cell(neighbor) and board.is_empty(neighbor):
                moves.append([coord, neighbor])

        return moves

    @staticmethod
    def _generate_jump_moves(board, current_coord, path):
        """递归生成跳跃移动"""
        moves = []

        # 检查所有可能的目标位置
        for target_coord in board.get_all_cells():
            if target_coord == current_coord or target_coord in path:
                continue

            # 检查是否是有效跳跃
            if ChineseCheckersMoves._is_valid_jump(board, current_coord, target_coord):
                new_path = path + [target_coord]
                moves.append(new_path)

                # 继续递归（连续跳跃）
                further_jumps = ChineseCheckersMoves._generate_jump_moves(board, target_coord, new_path)
                moves.extend(further_jumps)

        return moves

    @staticmethod
    def _is_valid_jump(board, from_coord, to_coord):
        """检查是否是从from到to的有效跳跃"""
        # 1. 目标必须为空
        if not board.is_empty(to_coord):
            return False

        # 2. 必须在一条直线上
        direction = from_coord.direction_to(to_coord)
        if direction is None:
            return False

        # 3. 距离必须大于1（跳跃至少跳过1个棋子）
        distance = from_coord.distance(to_coord)
<<<<<<< HEAD
        if distance < 2:
=======
        if distance !=2:
>>>>>>> 4df9e2bc9aeedfcc2694ab47d689d8b22fcbedb5
            return False

        # 4. 中间的所有格子都必须有棋子
        for step in range(1, distance):
            mid_cell = from_coord + (direction * step)
            if not board.is_valid_cell(mid_cell) or board.is_empty(mid_cell):
                return False

        return True

    @staticmethod
    def is_valid_move(board, move, player):
        """验证移动是否合法"""
        if len(move) < 2:
            return False

        # 检查起始位置是否有玩家的棋子
        start = move[0]
        if board.get_piece(start) != player:
            return False

        # 检查每个步骤是否合法
        for i in range(len(move) - 1):
            current = move[i]
            next_pos = move[i + 1]

            # 单步移动
            if current.is_neighbor(next_pos):
                if not board.is_empty(next_pos):
                    return False
            # 跳跃移动
            else:
                if not ChineseCheckersMoves._is_valid_jump(board, current, next_pos):
                    return False

        return True

    @staticmethod
    def apply_move(board, move):
        """在棋盘上执行移动"""
        new_board = board.copy()

        if len(move) < 2:
            return new_board

        # 移动棋子
        start = move[0]
        end = move[-1]
        piece = new_board.get_piece(start)

        new_board.set_piece(start, 0)
        new_board.set_piece(end, piece)

        return new_board

    @staticmethod
    def find_best_jump_path(board, start_coord, target_coord):
        """寻找从起点到目标的最佳跳跃路径（BFS）"""
        if start_coord == target_coord:
            return [start_coord]

        # 使用BFS搜索跳跃路径
        from collections import deque

        queue = deque()
        queue.append((start_coord, [start_coord]))  # (当前位置, 路径)
        visited = {start_coord}

        while queue:
            current, path = queue.popleft()

            # 尝试所有可能的跳跃
            for next_coord in board.get_all_cells():
                if next_coord in visited:
                    continue

                if ChineseCheckersMoves._is_valid_jump(board, current, next_coord):
                    new_path = path + [next_coord]

                    if next_coord == target_coord:
                        return new_path

                    visited.add(next_coord)
                    queue.append((next_coord, new_path))

        return None  # 没有找到路径