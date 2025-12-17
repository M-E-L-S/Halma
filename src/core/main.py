# chinese_checkers_gui.py
import pygame
import sys
import math
import os
from board import ChineseCheckersBoard, CubeCoord


class ChineseCheckersGUI:

    def __init__(self, board=None):
        pygame.init()

        # 棋盘参数
        self.board = board or ChineseCheckersBoard()

        # 颜色定义
        self.COLORS = {
            'bg': (240, 240, 240),
            'hex_center': (200, 230, 200),
            'grid': (100, 100, 100),
            'cell_border': (80, 80, 80),
            'player1': (255, 50, 50),  # 红色
            'player2': (50, 50, 255),  # 蓝色
            'highlight': (255, 255, 100),  # 高亮黄色
            'selected': (255, 200, 50),  # 选中橙色
            'text': (30, 30, 30),
            'region_tri0': (255, 200, 200),  # 玩家1区域（淡红）
            'region_tri3': (200, 200, 255),  # 玩家2区域（淡蓝）
            'region_hex': (230, 230, 200),  # 中心区域（淡黄）
        }

        # 窗口设置
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 900
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("中国跳棋 - Chinese Checkers")

        # 六边形几何参数
        self.hex_size = 30  # 六边形大小（内切圆半径）
        self.hex_height = self.hex_size * 2
        self.hex_width = math.sqrt(3) * self.hex_size

        # 视图偏移
        self.view_offset_x = self.SCREEN_WIDTH // 2
        self.view_offset_y = self.SCREEN_HEIGHT // 2

        # 游戏状态
        self.selected_piece = None
        self.valid_moves = []
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.message = ""
        self.message_timer = 0
        self.fonts_loaded = False
        self.font = None
        self.title_font = None
        self.status_font = None
        self._init_fonts()

        try:
            from moves import ChineseCheckersMoves
            self.moves_gen = ChineseCheckersMoves
        except ImportError:
            self.moves_gen = None

    def _init_fonts(self):
        try:
            font_path = "STSONG.TTF"
            if os.path.exists(font_path):
                self.font = pygame.font.Font(font_path, 24)
                self.title_font = pygame.font.Font(font_path, 36)
                self.status_font = pygame.font.Font(font_path, 28)
                self.fonts_loaded = True
        except Exception as e:
            self.font = pygame.font.SysFont(None, 24)
            self.title_font = pygame.font.SysFont(None, 36)
            self.status_font = pygame.font.SysFont(None, 28)

    def cube_to_pixel(self, coord):
        x = self.hex_size * (math.sqrt(3) * coord.q + math.sqrt(3) / 2 * coord.r)
        y = self.hex_size * (3 / 2 * coord.r)
        return (x + self.view_offset_x, y + self.view_offset_y)

    def pixel_to_cube(self, pixel_x, pixel_y):
        x = pixel_x - self.view_offset_x
        y = pixel_y - self.view_offset_y
        q = (math.sqrt(3) / 3 * x - 1 / 3 * y) / self.hex_size
        r = (2 / 3 * y) / self.hex_size
        q_round = round(q)
        r_round = round(r)
        s_round = round(-q - r)
        q_diff = abs(q_round - q)
        r_diff = abs(r_round - r)
        s_diff = abs(s_round - (-q - r))

        if q_diff > r_diff and q_diff > s_diff:
            q_round = -r_round - s_round
        elif r_diff > s_diff:
            r_round = -q_round - s_round
        else:
            s_round = -q_round - r_round

        return CubeCoord(q_round, r_round, s_round)

    def draw_hexagon(self, center, color, border_color=None, border_width=1):
        x, y = center
        points = []

        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            point_x = x + self.hex_size * math.cos(angle_rad)
            point_y = y + self.hex_size * math.sin(angle_rad)
            points.append((point_x, point_y))
        pygame.draw.polygon(self.screen, color, points)

        # 绘制边框
        if border_color:
            pygame.draw.polygon(self.screen, border_color, points, border_width)

    def draw_piece(self, center, player, selected=False):
        x, y = center

        # 棋子颜色
        if player == 1:
            color = self.COLORS['player1']
        elif player == -1:
            color = self.COLORS['player2']
        else:
            return

        radius = self.hex_size * 0.8
        pygame.draw.circle(self.screen, color, (x, y), int(radius))

        # 添加高光效果
        highlight_radius = radius * 0.7
        highlight_color = (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50))
        pygame.draw.circle(self.screen, highlight_color,
                           (int(x - radius * 0.3), int(y - radius * 0.3)),
                           int(highlight_radius * 0.4))

        # 如果被选中，绘制选中效果
        if selected:
            pygame.draw.circle(self.screen, self.COLORS['selected'], (x, y), int(radius * 1.1), 3)

    def get_cell_color(self, region):
        if region == 'tri0':
            return self.COLORS['region_tri0']
        elif region == 'tri3':
            return self.COLORS['region_tri3']
        elif region is not None and region.startswith('tri'):
            return (220, 220, 220)  # 其他三角形区域（灰色）
        elif region == 'hex':
            return self.COLORS['region_hex']
        else:
            return (255, 255, 255)  # 默认白色

    def draw_board(self):
        # 清屏
        self.screen.fill(self.COLORS['bg'])

        # 绘制所有格子
        for coord in self.board.get_all_cells():
            pixel_pos = self.cube_to_pixel(coord)
            region = self.board.get_region(coord)
            cell_color = self.get_cell_color(region)

            # 如果是有效移动目标，高亮显示
            is_valid_target = False
            if self.selected_piece and self.valid_moves:
                for move in self.valid_moves:
                    if len(move) > 0 and move[0] == self.selected_piece and move[-1] == coord:
                        is_valid_target = True
                        break

            # 调整颜色（如果是有效目标则变亮）
            if is_valid_target:
                highlight_factor = 1.3
                cell_color = (min(255, int(cell_color[0] * highlight_factor)),
                              min(255, int(cell_color[1] * highlight_factor)),
                              min(255, int(cell_color[2] * highlight_factor)))

            # 绘制六边形格子
            border_color = self.COLORS['cell_border']
            border_width = 1
            self.draw_hexagon(pixel_pos, cell_color, border_color, border_width)

            # 如果有棋子，绘制棋子
            piece = self.board.get_piece(coord)
            if piece != 0:
                # 修复：检查selected_piece是否为None
                is_selected = (self.selected_piece is not None and coord == self.selected_piece)
                self.draw_piece(pixel_pos, piece, is_selected)

    def draw_ui(self):
        # 绘制标题
        title = self.title_font.render("中国跳棋 - Chinese Checkers", True, self.COLORS['text'])
        self.screen.blit(title, (self.SCREEN_WIDTH // 2 - title.get_width() // 2, 10))

        # 绘制玩家信息
        player1_text = self.status_font.render("玩家1 (红色)", True, self.COLORS['player1'])
        player2_text = self.status_font.render("玩家2 (蓝色)", True, self.COLORS['player2'])
        self.screen.blit(player1_text, (50, 60))
        self.screen.blit(player2_text, (self.SCREEN_WIDTH - 200, 60))

        # 绘制当前玩家
        current_player_str = f"当前回合: {'玩家1' if self.current_player == 1 else '玩家2'}"
        current_player_text = self.status_font.render(
            current_player_str,
            True,
            self.COLORS['player1'] if self.current_player == 1 else self.COLORS['player2']
        )
        self.screen.blit(current_player_text, (self.SCREEN_WIDTH // 2 - current_player_text.get_width() // 2, 60))

        # 绘制棋子统计
        player1_pieces = len(self.board.get_player_pieces(1))
        player2_pieces = len(self.board.get_player_pieces(-1))

        pieces_text1 = self.font.render(f"玩家1棋子: {player1_pieces}/10", True, self.COLORS['text'])
        pieces_text2 = self.font.render(f"玩家2棋子: {player2_pieces}/10", True, self.COLORS['text'])
        self.screen.blit(pieces_text1, (50, 90))
        self.screen.blit(pieces_text2, (self.SCREEN_WIDTH - 200, 90))

        # 绘制进度
        for player in [1, -1]:
            target_region = self.board.player_target_regions[player]
            player_pieces = self.board.get_player_pieces(player)
            in_target = sum(1 for coord in player_pieces
                            if self.board.get_region(coord) == target_region)

            progress_str = f"目标区域: {in_target}/10"
            progress_text = self.font.render(progress_str, True, self.COLORS['text'])

            if player == 1:
                self.screen.blit(progress_text, (50, 120))
            else:
                self.screen.blit(progress_text, (self.SCREEN_WIDTH - 200, 120))

        # 绘制操作说明
        instructions = [
            "操作说明:",
            "1. 点击棋子选择",
            "2. 点击目标格子移动",
            "3. ESC: 取消选择",
            "4. R: 重新开始游戏",
            "5. 空格: 随机视角",
            "6. C: 居中视角"
        ]

        for i, text in enumerate(instructions):
            instruction = self.font.render(text, True, self.COLORS['text'])
            self.screen.blit(instruction, (50, self.SCREEN_HEIGHT - 180 + i * 25))

        # 如果游戏结束，显示获胜者
        if self.game_over and self.winner:
            winner_color = self.COLORS['player1'] if self.winner == 1 else self.COLORS['player2']
            winner_str = f"游戏结束! 玩家{self.winner} 获胜!"
            winner_text = self.title_font.render(winner_str, True, winner_color)
            text_rect = winner_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            pygame.draw.rect(self.screen, self.COLORS['bg'], text_rect.inflate(20, 10), border_radius=10)
            pygame.draw.rect(self.screen, winner_color, text_rect.inflate(20, 10), 3, border_radius=10)
            self.screen.blit(winner_text, text_rect)

        # 显示消息
        if self.message and self.message_timer > 0:
            message_text = self.font.render(self.message, True, (255, 50, 50))
            self.screen.blit(message_text, (self.SCREEN_WIDTH // 2 - message_text.get_width() // 2, 150))
            self.message_timer -= 1

    def show_message(self, message, duration=60):
        self.message = message
        self.message_timer = duration

    def handle_click(self, pos):
        if self.game_over:
            return

        x, y = pos
        clicked_coord = self.pixel_to_cube(x, y)
        if not self.board.is_valid_cell(clicked_coord):
            self.show_message("无效的格子!")
            return
        if self.selected_piece is not None:
            piece = self.board.get_piece(clicked_coord)
            if piece == self.current_player:
                self.selected_piece = clicked_coord
                self.update_valid_moves()
                return
            for move in self.valid_moves:
                if len(move) > 0 and move[0] == self.selected_piece and move[-1] == clicked_coord:
                    self.execute_move(move)
                    return
            self.selected_piece = None
            self.valid_moves = []
            self.show_message("无效的移动!")

        else:
            piece = self.board.get_piece(clicked_coord)
            if piece == self.current_player:
                self.selected_piece = clicked_coord
                self.update_valid_moves()
            elif piece != 0:
                self.show_message("这不是你的棋子!")

    def update_valid_moves(self):
        self.valid_moves = []

        if not self.selected_piece or not self.moves_gen:
            return
        self.valid_moves = self.moves_gen.generate_moves_for_piece(self.board, self.selected_piece)
        if self.board.get_piece(self.selected_piece) != self.current_player:
            self.selected_piece = None
            self.valid_moves = []

    def execute_move(self, move):
        try:
            self.board = self.moves_gen.apply_move(self.board, move)
            self.current_player *= -1
            self.selected_piece = None
            self.valid_moves = []
            self.check_game_over()
            move_info = f"移动了 {len(move) - 1} 步"
            if len(move) > 2:
                move_info += f" (包含{len(move) - 2}次跳跃)"
            self.show_message(move_info)

        except Exception as e:
            self.show_message(f"移动错误: {str(e)}")
            print(f"移动错误: {e}")

    def check_game_over(self):
        # 检查玩家1是否获胜
        player1_won = True
        target_region = self.board.player_target_regions[1]
        player1_pieces = self.board.get_player_pieces(1)

        if len(player1_pieces) != 10:
            player1_won = False
        else:
            for coord in player1_pieces:
                if self.board.get_region(coord) != target_region:
                    player1_won = False
                    break

        # 检查玩家2是否获胜
        player2_won = True
        target_region = self.board.player_target_regions[-1]
        player2_pieces = self.board.get_player_pieces(-1)

        if len(player2_pieces) != 10:
            player2_won = False
        else:
            for coord in player2_pieces:
                if self.board.get_region(coord) != target_region:
                    player2_won = False
                    break

        if player1_won:
            self.game_over = True
            self.winner = 1
        elif player2_won:
            self.game_over = True
            self.winner = -1

    def reset_game(self):
        self.board = ChineseCheckersBoard()
        self.selected_piece = None
        self.valid_moves = []
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.message = ""
        self.message_timer = 0
        self.view_offset_x = self.SCREEN_WIDTH // 2
        self.view_offset_y = self.SCREEN_HEIGHT // 2

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键点击
                        self.handle_click(event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # 取消选择
                        self.selected_piece = None
                        self.valid_moves = []

                    elif event.key == pygame.K_r:
                        # 重新开始游戏
                        self.reset_game()

                    elif event.key == pygame.K_SPACE:
                        # 随机视角
                        import random
                        self.view_offset_x = random.randint(200, 1000)
                        self.view_offset_y = random.randint(200, 700)

                    elif event.key == pygame.K_c:
                        # 居中视角
                        self.view_offset_x = self.SCREEN_WIDTH // 2
                        self.view_offset_y = self.SCREEN_HEIGHT // 2

            # 绘制游戏
            self.draw_board()
            self.draw_ui()

            # 更新显示
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    # 创建并运行游戏
    game = ChineseCheckersGUI()
    game.run()


if __name__ == "__main__":
    main()