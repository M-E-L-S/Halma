# moves.py
from sympy import false

from src.core.board import *
class ChineseCheckersMoves:
    """中国跳棋移动生成器"""
    @staticmethod
    def remove_duplicate_moves(all_moves):
        """删除重复的移动路径 - 只比较起点和终点"""
        if not all_moves:
        #    print("调试: 输入为空列表")
            return []

     #   print(f"调试: 开始处理 {len(all_moves)} 条移动")

        unique_moves = []
        seen_keys = set()  # 存储已经见过的起点-终点对
        duplicate_count = 0

        for i, move in enumerate(all_moves):
            if len(move) < 2:
           #     print(f"调试: 第{i}条移动无效，长度={len(move)}")
                continue  # 跳过无效的移动

            start = move[0]
            end = move[-1]

            # 创建唯一的键：起点和终点坐标
            key = (start.q, start.r, start.s, end.q, end.r, end.s)

            # 调试信息
          #  start_str = f"({start.q},{start.r},{start.s})"
       #     end_str = f"({end.q},{end.r},{end.s})"
         #   move_type = "单步" if len(move) == 2 else f"跳跃({len(move) - 1}步)"

            if key not in seen_keys:
            #    print(f"调试: 第{i}条移动 - {move_type} {start_str} -> {end_str} [新]")
                seen_keys.add(key)
                unique_moves.append(move)
            else:
          #      print(f"调试: 第{i}条移动 - {move_type} {start_str} -> {end_str} [重复]")
                duplicate_count += 1

      #  print(f"调试: 处理完成，原始{len(all_moves)}条，去重后{len(unique_moves)}条")
       # print(f"调试: 删除了 {duplicate_count} 条重复移动")

        # 显示最终的唯一移动
      #  print("\n调试: 最终的唯一移动:")
        for i, move in enumerate(unique_moves):
            start = move[0]
            end = move[-1]
            move_type = "单步" if len(move) == 2 else f"跳跃({len(move) - 1}步)"
      #      print(f"  {i}: {move_type} ({start.q},{start.r},{start.s}) -> ({end.q},{end.r},{end.s})")

        return unique_moves

    @staticmethod
    def generate_all_moves(board, player):
        """生成玩家所有合法移动"""
        all_moves = []
        player_pieces = board.get_player_pieces(player)

        for start_coord in player_pieces:
            piece_moves = ChineseCheckersMoves.generate_moves_for_piece(board, start_coord)
            all_moves.extend(piece_moves)
        result = ChineseCheckersMoves.remove_duplicate_moves(all_moves)
        for move in result:
            coord = move[0]
            neighbor = move[-1]
       #     print("move:", coord.q, coord.r, coord.s, "to", neighbor.q, neighbor.r, neighbor.s)
        return result



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
        for move in moves:
            coord = move[0]
            neighbor = move[-1]
      #      print("amove:", coord.q, coord.r, coord.s, "to", neighbor.q, neighbor.r, neighbor.s)
        return moves

    @staticmethod
    def _generate_single_moves(board, coord):
        """生成单步移动"""
        moves = []

        for direction in range(6):
            neighbor = coord.neighbor(direction)
            if board.is_valid_cell(neighbor) and board.is_empty(neighbor):
                moves.append([coord, neighbor])
       #         print("single_move:",coord.q,coord.r,coord.s,"to",neighbor.q,neighbor.r,neighbor.s)

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
                if new_path not in moves:
                    moves.append(new_path)

                # 继续递归（连续跳跃）
                further_jumps = ChineseCheckersMoves._generate_jump_moves(board, target_coord, new_path)
                for further_jump in further_jumps:
                    if further_jump not in moves:
                        moves.append(further_jump)
        for move in moves:
            coord = move[0]
            neighbor = move[1]
     #       print("jump_move:", coord.q, coord.r, coord.s, "to", neighbor.q, neighbor.r, neighbor.s)
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

        # 3. 普通模式下距离必须为2
        distance = from_coord.distance(to_coord)
        current_mode = board.get_current_mode()
        if (current_mode == False) and (distance !=2):
            return False

        # 4. 普通模式下中间的所有格子都必须有棋子
        if not current_mode:
            for step in range(1, distance):
                mid_cell = from_coord + (direction * step)
                if not board.is_valid_cell(mid_cell) or board.is_empty(mid_cell):
                    return False
        # 5. 镜像模式下正中为棋子，其余为空
        if current_mode:
            if distance % 2 != 0:
                return False
            mid_cell = from_coord + (direction * distance // 2)
            if not board.is_valid_cell(mid_cell) or board.is_empty(mid_cell):
                return False
            for step in range(1, distance):
                cell = from_coord + (direction * step)
                if cell == mid_cell:
                    continue
                if not board.is_empty(cell):
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