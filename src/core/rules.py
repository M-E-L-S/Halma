# rules.py
from moves import *
class GameRules:
    @staticmethod
    def get_all_valid_moves(board, player):
        """获取玩家所有合法走法"""
        all_moves = []

        # 优先：检查是否有强制吃子
        capture_moves = []
        simple_moves = []

        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 1 and board.board[i][j] * player > 0:
                    # 获取该棋子的所有可能走法
                    captures = get_capture_moves(board, (i, j))
                    if captures:
                        capture_moves.extend(captures)
                    else:
                        simple_moves.extend(get_simple_moves(board, (i, j)))

        # 跳棋规则：有吃子必须吃
        if capture_moves:
            return capture_moves
        else:
            return simple_moves

    @staticmethod
    def is_game_over(board):
        """检查游戏是否结束"""
        # 检查双方是否还有棋子
        player1_pieces = 0
        player2_pieces = 0

        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    if board.board[i][j] > 0:
                        player1_pieces += 1
                    elif board.board[i][j] < 0:
                        player2_pieces += 1

        return player1_pieces == 0 or player2_pieces == 0