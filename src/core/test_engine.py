# test_chinese_checkers.py - 修正版本
def test_chinese_checkers():
    """测试中国跳棋引擎"""
    print("测试中国跳棋引擎...")
    print("=" * 50)

<<<<<<< HEAD
    from src.core.board import ChineseCheckersBoard
    from src.core.moves import ChineseCheckersMoves
    from src.core.game_state import ChineseCheckersGame
    from src.core.board import CubeCoord
=======
    from board import ChineseCheckersBoard
    from moves import ChineseCheckersMoves
    from game_state import ChineseCheckersGame
    from board import CubeCoord
>>>>>>> 4df9e2bc9aeedfcc2694ab47d689d8b22fcbedb5

    # 1. 创建棋盘
    print("1. 创建棋盘...")
    board = ChineseCheckersBoard()
    board.print_board_info()

    # 2. 测试坐标系统
    print("\n2. 测试坐标系统...")
    coord1 = CubeCoord(0, 0, 0)
    coord2 = CubeCoord(2, -1, -1)  # 东方向两个单位

    print(f"坐标1: {coord1}")
    print(f"坐标2: {coord2}")
    print(f"距离: {coord1.distance(coord2)}")
    print(f"方向向量: {coord1.direction_to(coord2)}")
    print(f"是否直线: {coord1.is_straight_line_to(coord2)}")

    # 3. 测试移动生成
    print("\n3. 测试移动生成...")
    player1_pieces = board.get_player_pieces(1)
    if player1_pieces:
        test_piece = player1_pieces[0]
        print(f"测试棋子位置: {test_piece}")

        moves = ChineseCheckersMoves.generate_moves_for_piece(board, test_piece)
        print(f"该棋子的可能移动: {len(moves)} 种")

        if moves:
            print("前3个移动:")
            for i, move in enumerate(moves[:3]):
                print(f"  {i + 1}. {move}")

    # 4. 测试跳跃逻辑
    print("\n4. 测试跳跃逻辑...")
    test_board = ChineseCheckersBoard()

    # 清空棋盘
    for coord in test_board.cells:
        test_board.cells[coord] = 0

    # 设置一个简单的跳跃测试
    A = CubeCoord(0, 0, 0)
    B = CubeCoord(1, 0, -1)  # 东方向邻居
    C = CubeCoord(2, 0, -2)  # 东方向第二个
    D = CubeCoord(3, 0, -3)  # 东方向第三个

    test_board.set_piece(A, 1)  # 玩家棋子
    test_board.set_piece(B, -1)  # 任意棋子（作为跳板）
    test_board.set_piece(C, -1)  # 另一个棋子

    print(f"测试局面:")
    print(f"  A {A}: 玩家棋子")
    print(f"  B {B}: 跳板棋子")
    print(f"  C {C}: 跳板棋子")
    print(f"  D {D}: 空位")

    # 测试跳跃
    print(f"\n从A到C是否有效跳跃: {ChineseCheckersMoves._is_valid_jump(test_board, A, C)}")
    print(f"从A到D是否有效跳跃: {ChineseCheckersMoves._is_valid_jump(test_board, A, D)}")

    # 生成跳跃移动
    jump_moves = ChineseCheckersMoves._generate_jump_moves(test_board, A, [A])
    print(f"从A的跳跃移动: {len(jump_moves)} 种")
    for move in jump_moves:
        print(f"  {move}")

    # 5. 创建完整游戏
    print("\n5. 创建完整游戏...")
    game = ChineseCheckersGame()
    game.print_status()

    # 6. 测试游戏移动
    state = game.get_state()
    if state['valid_moves']:
        print(f"\n6. 测试执行移动...")
        print(f"可用移动数: {len(state['valid_moves'])}")

        # 找一个跳跃移动（如果有）
        jump_move = None
        for move in state['valid_moves']:
            if len(move) > 2 or (len(move) == 2 and not move[0].is_neighbor(move[1])):
                jump_move = move
                break

        if jump_move:
            print(f"测试跳跃移动: {jump_move}")
            success, message = game.make_move(jump_move)
            print(f"结果: {success}, 消息: {message}")
            game.print_status()
        else:
            print("没有找到跳跃移动，测试单步移动")
            test_move = state['valid_moves'][0]
            print(f"测试移动: {test_move}")
            success, message = game.make_move(test_move)
            print(f"结果: {success}, 消息: {message}")
            game.print_status()

    print("\n" + "=" * 50)
    print("中国跳棋引擎测试完成！")


if __name__ == "__main__":
    test_chinese_checkers()