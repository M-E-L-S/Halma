# game_state.py
from board import ChineseCheckersBoard
from moves import ChineseCheckersMoves

class ChineseCheckersGame:
    """中国跳棋游戏状态管理"""

    def __init__(self, board=None):
        self.board = board or ChineseCheckersBoard()
        self.current_player = 1 # 玩家1先走
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.turn_count = 0


    def get_state(self):
        """获取当前游戏状态（给AI使用）"""
        return {
            'board': self.board,
            'board_cells': self.board.cells.copy(),
            'current_player': self.current_player,
            'valid_moves': ChineseCheckersMoves.generate_all_moves(self.board, self.current_player),
            'game_over': self.game_over,
            'winner': self.winner,
            'turn_count': self.turn_count,
            'player_progress': self._get_player_progress(),
            'special_mode': self.board.get_current_mode()
        }

    def _get_player_progress(self):
        """获取玩家进度"""
        progress = {}
        for player in [1, -1]:
            target_region = self.board.player_target_regions[player]
            player_pieces = self.board.get_player_pieces(player)

            in_target = sum(1 for coord in player_pieces
                            if self.board.get_region(coord) == target_region)

            progress[player] = {
                'total_pieces': len(player_pieces),
                'in_target': in_target,
                'percentage': in_target / 10 if len(player_pieces) > 0 else 0
            }

        return progress

    def make_move(self, move):
        """执行移动"""
        if self.game_over:
            return False, "游戏已结束"

        # 验证移动合法性
        valid_moves = ChineseCheckersMoves.generate_all_moves(self.board, self.current_player)

        # 标准化移动表示以便比较
        move_tuple = tuple(move)
        valid_moves_tuples = [tuple(m) for m in valid_moves]

        if move_tuple not in valid_moves_tuples:
            return False, "非法移动"

        # 执行移动
        self.board = ChineseCheckersMoves.apply_move(self.board, move)

        # 更新状态
        self.move_history.append(move)
        self.current_player *= -1  # 切换玩家
        self.turn_count += 1

        # 检查游戏是否结束
        self._check_game_over()

        return True, "移动成功"

    def _check_game_over(self):
        """检查游戏是否结束"""
        # 检查玩家1是否获胜
        if self._player_has_won(1):
            self.game_over = True
            self.winner = 1
            return

        # 检查玩家2是否获胜
        if self._player_has_won(-1):
            self.game_over = True
            self.winner = -1
            return

    def _player_has_won(self, player):
        """检查玩家是否获胜"""
        target_region = self.board.player_target_regions[player]
        player_pieces = self.board.get_player_pieces(player)

        # 所有棋子都必须在目标区域
        if len(player_pieces) != 10:  # 应该是10个棋子
            return False

        for coord in player_pieces:
            if self.board.get_region(coord) != target_region:
                return False

        return True

    def get_winner(self):
        """获取获胜者"""
        if not self.game_over:
            return None

        return self.winner

    def reset(self):
        """重置游戏"""
        self.__init__()

    def print_status(self):
        """打印游戏状态"""
        print(f"\n=== 回合 {self.turn_count} ===")
        print(f"当前玩家: {'玩家1' if self.current_player == 1 else '玩家2'}")

        state = self.get_state()
        print(f"可用移动数: {len(state['valid_moves'])}")

        progress = state['player_progress']
        for player in [1, -1]:
            p = progress[player]
            print(f"玩家{player}: {p['in_target']}/10 棋子在目标区域 ({p['percentage'] * 100:.1f}%)")

        if self.game_over:
            print(f"\n游戏结束！获胜者: 玩家{self.winner}")