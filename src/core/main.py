# chinese_checkers_gui.py - 修正完整版本
import pygame
import sys
import math
import os
from src.core.board import ChineseCheckersBoard, CubeCoord
from src.ai.seacrch import Search

class ChineseCheckersGUI:
    """中国跳棋图形界面"""

    def __init__(self, board=None):
        self.all_moves = None
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
            'gender':(85,242,242),
            'region_tri0': (255, 200, 200),  # 玩家1区域（淡红）
            'region_tri3': (200, 200, 255),  # 玩家2区域（淡蓝）
            'region_hex': (230, 230, 200),  # 中心区域（淡黄）
            'button_normal': (100, 150, 200),  # 按钮正常状态
            'button_hover': (120, 170, 220),  # 按钮悬停状态
            'button_active': (80, 130, 180),  # 按钮激活状态
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

        # 按钮定义
        self.buttons = {
            'mode_toggle': {
                'rect': pygame.Rect(self.SCREEN_WIDTH - 200, 250, 160, 40),
                'text': '切换到镜像模式',
                'active': False,
                'hover': False
            },
            'restart': {
                'rect': pygame.Rect(self.SCREEN_WIDTH - 200, 300, 160, 40),
                'text': '重新开始游戏',
                'active': False,
                'hover': False
            },
            'center_view': {
                'rect': pygame.Rect(self.SCREEN_WIDTH - 200, 350, 160, 40),
                'text': '修复视角偏移',
                'active': False,
                'hover': False
            },
            'ai_toggle':{
                'rect': pygame.Rect(self.SCREEN_WIDTH - 200, 400, 160, 40),
                'text': '玩家2设为AI',
                'active': False,
                'hover': False
            }
        }

        # 字体初始化
        self._init_fonts()

        # 从moves导入移动生成器
        try:
            from moves import ChineseCheckersMoves
            self.moves_gen = ChineseCheckersMoves
            print("移动生成器加载成功")
        except ImportError as e:
            print(f"警告: 无法导入移动生成器: {e}")
            self.moves_gen = None

        # 更新按钮文本
        self._update_mode_button_text()

        # 初始化棋盘模式
        if hasattr(self.board, 'special_mode'):
            print(f"初始模式: {'镜像模式' if self.board.special_mode else '普通模式'}")
        else:
            print("警告: 棋盘没有special_mode属性")
        # 添加AI设置
        self.ai_players = {}  # 存储AI玩家
        self.use_ai = False  # 是否启用AI
        self.ai_thinking = False  # AI是否正在思考

    def init_ai(self, player_id, depth=3):
        """初始化AI玩家"""
        try:
            from src.ai.ai_agent import AIPlayer
            self.ai_players[player_id] = AIPlayer(player_id, depth)
            print(f"AI玩家{player_id}初始化完成 (搜索深度: {depth})")
        except ImportError as e:
            print(f"无法导入AI模块: {e}")
            return False
        return True

    def toggle_ai_mode(self, player_id=None):
        """切换AI模式"""
        if not self.ai_players:
            # 如果没有初始化AI，先初始化
            if player_id is None:
                player_id = -1  # 默认玩家2为AI
            if self.init_ai(player_id):
                self.use_ai = True
                self.show_message(f"玩家{player_id}已设为AI")

                # 如果是当前玩家是AI，立即行动
                if self.current_player == player_id and not self.game_over:
                    self.ai_move()
            else:
                self.show_message("AI初始化失败")
        else:
            # 移除AI
            self.ai_players.clear()
            self.use_ai = False
            self.show_message("AI模式已关闭")

    def ai_move(self):
        """执行AI移动"""
        if not self.use_ai or self.game_over or self.ai_thinking:
            return

        if self.current_player in self.ai_players:
            self.ai_thinking = True
            self.show_message("AI正在思考...")

            # 获取游戏状态
            from game_state import ChineseCheckersGame
            game_state_obj = ChineseCheckersGame(self.board)
            game_state = game_state_obj.get_state()

            # 获取AI移动
            ai_player = self.ai_players[self.current_player]
            best_move = ai_player.get_move(game_state)

            if best_move:
                # 执行移动
                self.execute_move(best_move)
            else:
                self.show_message("AI没有找到合法移动")
                self.current_player *= -1  # 切换玩家

            self.ai_thinking = False

    def _init_fonts(self):
        """初始化字体"""
        try:
            font_paths = [
                "STXINGKA.TTF",
                "./STXINGKA.TTF",
                "../STXINGKA.TTF",
                "C:/Windows/Fonts/STXINGKA.TTF",
                "simsun.ttc",
            ]

            font_loaded = False
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        self.font = pygame.font.Font(path, 24)
                        self.title_font = pygame.font.Font(path, 36)
                        self.status_font = pygame.font.Font(path, 28)
                        self.button_font = pygame.font.Font(path, 20)
                        font_loaded = True
                        print(f"使用字体文件: {path}")
                        break
                    except:
                        continue

            if not font_loaded:
                self.font = pygame.font.SysFont(None, 24)
                self.title_font = pygame.font.SysFont(None, 36)
                self.status_font = pygame.font.SysFont(None, 28)
                self.button_font = pygame.font.SysFont(None, 20)
                print("使用系统默认字体")

        except Exception as e:
            print(f"字体初始化错误: {e}")
            self.font = pygame.font.SysFont(None, 24)
            self.title_font = pygame.font.SysFont(None, 36)
            self.status_font = pygame.font.SysFont(None, 28)
            self.button_font = pygame.font.SysFont(None, 20)

    def _update_mode_button_text(self):
        """更新模式切换按钮的文本"""
        try:
            # 尝试获取当前模式
            if hasattr(self.board, 'get_current_mode') and callable(self.board.get_current_mode):
                current_mode = self.board.get_current_mode()
            elif hasattr(self.board, 'special_mode'):
                current_mode = self.board.special_mode
            else:
                current_mode = False

            if current_mode:
                self.buttons['mode_toggle']['text'] = '切换到普通模式'
            else:
                self.buttons['mode_toggle']['text'] = '切换到镜像模式'
        except Exception as e:
            print(f"更新按钮文本错误: {e}")
            self.buttons['mode_toggle']['text'] = '切换模式'

    def cube_to_pixel(self, coord):
        """将立方坐标转换为像素坐标"""
        x = self.hex_size * (math.sqrt(3) * coord.q + math.sqrt(3) / 2 * coord.r)
        y = self.hex_size * (3 / 2 * coord.r)
        return (x + self.view_offset_x, y + self.view_offset_y)

    def pixel_to_cube(self, pixel_x, pixel_y):
        """将像素坐标转换为立方坐标（近似）"""
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
        """绘制一个六边形"""
        x, y = center
        points = []

        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            point_x = x + self.hex_size * math.cos(angle_rad)
            point_y = y + self.hex_size * math.sin(angle_rad)
            points.append((point_x, point_y))

        pygame.draw.polygon(self.screen, color, points)

        if border_color:
            pygame.draw.polygon(self.screen, border_color, points, border_width)

    def draw_piece(self, center, player, selected=False):
        """绘制棋子"""
        x, y = center

        if player == 1:
            color = self.COLORS['player1']
        elif player == -1:
            color = self.COLORS['player2']
        else:
            return

        radius = self.hex_size * 0.8
        pygame.draw.circle(self.screen, color, (x, y), int(radius))

        highlight_radius = radius * 0.7
        highlight_color = (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50))
        pygame.draw.circle(self.screen, highlight_color,
                           (int(x - radius * 0.3), int(y - radius * 0.3)),
                           int(highlight_radius * 0.4))

        if selected:
            pygame.draw.circle(self.screen, self.COLORS['selected'], (x, y), int(radius * 1.1), 3)

    def get_cell_color(self, region):
        """根据区域类型获取格子颜色"""
        if region == 'tri0':
            return self.COLORS['region_tri0']
        elif region == 'tri3':
            return self.COLORS['region_tri3']
        elif region is not None and region.startswith('tri'):
            return (220, 220, 220)
        elif region == 'hex':
            return self.COLORS['region_hex']
        else:
            return (255, 255, 255)

    def draw_board(self):
        """绘制整个棋盘"""
        self.screen.fill(self.COLORS['bg'])

        for coord in self.board.get_all_cells():
            pixel_pos = self.cube_to_pixel(coord)
            region = self.board.get_region(coord)
            cell_color = self.get_cell_color(region)

            is_valid_target = False
            if self.selected_piece and self.valid_moves:
                for move in self.valid_moves:
                    if len(move) > 0 and move[0] == self.selected_piece and move[-1] == coord:
                        is_valid_target = True
                        break

            if is_valid_target:
                highlight_factor = 1.3
                cell_color = (min(255, int(cell_color[0] * highlight_factor)),
                              min(255, int(cell_color[1] * highlight_factor)),
                              min(255, int(cell_color[2] * highlight_factor)))

            border_color = self.COLORS['cell_border']
            border_width = 1
            self.draw_hexagon(pixel_pos, cell_color, border_color, border_width)

            piece = self.board.get_piece(coord)
            if piece != 0:
                is_selected = (self.selected_piece is not None and coord == self.selected_piece)
                self.draw_piece(pixel_pos, piece, is_selected)

    def draw_button(self, button_key):
        """绘制按钮"""
        button = self.buttons[button_key]
        rect = button['rect']

        if button['active']:
            color = self.COLORS['button_active']
        elif button['hover']:
            color = self.COLORS['button_hover']
        else:
            color = self.COLORS['button_normal']

        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLORS['gender'], rect, 5, border_radius=8)

        text_surface = self.button_font.render(button['text'], True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_ui(self):
        """绘制用户界面元素"""
        # 绘制标题
        title = self.title_font.render("中国跳棋 - Chinese Checkers", True, self.COLORS['text'])
        self.screen.blit(title, (self.SCREEN_WIDTH // 2 - title.get_width() // 2, 10))

        # 绘制玩家信息
        player1_text = self.status_font.render("玩家1 (红色)", True, self.COLORS['player1'])
        player2_text = self.status_font.render("玩家2 (蓝色)", True, self.COLORS['player2'])
        self.screen.blit(player1_text, (50, 60))
        self.screen.blit(player2_text, (self.SCREEN_WIDTH - 250, 60))

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
        self.screen.blit(pieces_text2, (self.SCREEN_WIDTH - 250, 90))

        # 绘制当前模式
        try:
            if hasattr(self.board, 'get_current_mode') and callable(self.board.get_current_mode):
                current_mode = self.board.get_current_mode()
            elif hasattr(self.board, 'special_mode'):
                current_mode = self.board.special_mode
            else:
                current_mode = False

            mode_str = "当前模式: 镜像模式" if current_mode else "当前模式: 普通模式"
        except:
            mode_str = "当前模式: 普通模式"

        mode_text = self.status_font.render(mode_str, True, self.COLORS['text'])
        self.screen.blit(mode_text, (self.SCREEN_WIDTH - 250, 180))

        # 绘制按钮
        for button_key in self.buttons:
            self.draw_button(button_key)

        # 绘制操作说明
        instructions = [
            "操作说明:",
            "1. 点击棋子选择",
            "2. 点击目标格子移动",
            "3. ESC: 取消选择",
            "4. 空格: 随机视角",

        ]

        for i, text in enumerate(instructions):
            instruction = self.font.render(text, True, self.COLORS['text'])
            self.screen.blit(instruction, (50, self.SCREEN_HEIGHT - 180 + i * 25))

        # 绘制模式规则说明
        try:
            if hasattr(self.board, 'get_current_mode') and callable(self.board.get_current_mode):
                current_mode = self.board.get_current_mode()
            elif hasattr(self.board, 'special_mode'):
                current_mode = self.board.special_mode
            else:
                current_mode = False

            if current_mode:
                mode_instructions = [
                    "镜像模式规则:",
                    "1. 跳跃距离必须为偶数",
                    "2. 中间正中的格子必须有棋子",
                    "3. 其他中间格子必须为空"
                ]
            else:
                mode_instructions = [
                    "普通模式规则:",
                    "1. 跳跃距离必须为2",
                    "2. 中间格子必须有棋子"
                ]
        except:
            mode_instructions = ["模式信息加载失败"]

        for i, text in enumerate(mode_instructions):
            instruction = self.font.render(text, True, self.COLORS['text'])
            self.screen.blit(instruction, (self.SCREEN_WIDTH - 350, self.SCREEN_HEIGHT - 180 + i * 25))

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
        """显示消息"""
        self.message = message
        self.message_timer = duration

    def handle_click(self, pos):
        """处理鼠标点击"""
        if self.game_over or self.ai_thinking:
            return

            # 如果当前玩家是AI，不接受玩家输入
        if self.use_ai and self.current_player in self.ai_players:
            self.show_message("现在是AI的回合，请等待")
            return

        x, y = pos

        # 检查是否点击了按钮
        for button_key, button in self.buttons.items():
            if button['rect'].collidepoint(x, y):
                self.handle_button_click(button_key)
                return

        # 如果不是点击按钮，处理棋盘点击
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

    def handle_button_click(self, button_key):
        """处理按钮点击"""
        if button_key == 'ai_toggle':
            if self.use_ai:
                self.toggle_ai_mode()
                self.buttons['ai_toggle']['text'] = '玩家2设为AI'
            else:
                self.toggle_ai_mode(-1)
                self.buttons['ai_toggle']['text'] = '关闭AI模式'
        if button_key == 'mode_toggle':
            # 切换游戏模式
            try:
                if hasattr(self.board, 'change_mode') and callable(self.board.change_mode):
                    self.board.change_mode()
                elif hasattr(self.board, 'special_mode'):
                    self.board.special_mode = not self.board.special_mode
                else:
                    self.show_message("无法切换模式")
                    return

                self._update_mode_button_text()
                self.selected_piece = None
                self.valid_moves = []

                # 获取当前模式名称
                if hasattr(self.board, 'get_current_mode') and callable(self.board.get_current_mode):
                    current_mode = self.board.get_current_mode()
                elif hasattr(self.board, 'special_mode'):
                    current_mode = self.board.special_mode
                else:
                    current_mode = False

                mode_name = "镜像模式" if current_mode else "普通模式"
                self.show_message(f"已切换到{mode_name}")

            except Exception as e:
                self.show_message(f"切换模式失败: {str(e)}")
                print(f"切换模式错误: {e}")

        elif button_key == 'restart':
            self.reset_game()
            self.show_message("游戏已重新开始")

        elif button_key == 'center_view':
            self.view_offset_x = self.SCREEN_WIDTH // 2
            self.view_offset_y = self.SCREEN_HEIGHT // 2
            self.show_message("视角已居中")

    def update_button_hover(self, pos):
        """更新按钮悬停状态"""
        x, y = pos
        for button_key, button in self.buttons.items():
            button['hover'] = button['rect'].collidepoint(x, y)

    def update_button_active(self, button_key, active):
        """更新按钮激活状态"""
        self.buttons[button_key]['active'] = active

    def update_valid_moves(self):
        """更新当前选中棋子的有效移动"""
        self.valid_moves = []

        if not self.selected_piece or not self.moves_gen:
            return

        try:
            self.valid_moves = self.moves_gen.generate_moves_for_piece(self.board, self.selected_piece)

            if self.board.get_piece(self.selected_piece) != self.current_player:
                self.selected_piece = None
                self.valid_moves = []
        except Exception as e:
            print(f"生成移动错误: {e}")
            self.valid_moves = []

    def execute_move(self, move):
        """执行移动"""
        try:
            if not self.moves_gen:
                self.show_message("移动功能未初始化")
                return

            # 验证移动
            if not self.moves_gen.is_valid_move(self.board, move, self.current_player):
                self.show_message("非法移动")
                return

            # 应用移动
            self.board = self.moves_gen.apply_move(self.board, move)

            # 切换玩家
            self.current_player *= -1

            # 清除选择
            self.selected_piece = None
            self.valid_moves = []

            # 检查游戏是否结束
            self.check_game_over()

            # 显示移动信息
            move_info = f"移动了 {len(move) - 1} 步"
            self.all_moves = self.moves_gen.generate_all_moves(self.board, 1)
            if len(move) > 2:
                move_info += f" (包含{len(move) - 2}次跳跃)"
            self.show_message(move_info)

            # 如果启用了AI，检查是否需要AI行动
            if self.use_ai and not self.game_over:
                # 延迟一小段时间让玩家看到移动效果
                import time
                pygame.time.delay(300)
                self.ai_move()
        except Exception as e:
            self.show_message(f"移动错误: {str(e)}")
            print(f"移动错误: {e}")

    def check_game_over(self):
        """检查游戏是否结束"""
        try:
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

        except Exception as e:
            print(f"检查游戏结束错误: {e}")

    def reset_game(self):
        """重置游戏"""
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

        for button_key in self.buttons:
            self.buttons[button_key]['active'] = False
            self.buttons[button_key]['hover'] = False

        self._update_mode_button_text()

    def run(self):
        """运行主游戏循环"""
        clock = pygame.time.Clock()
        running = True

        while running:
            mouse_pos = pygame.mouse.get_pos()
            self.update_button_hover(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                        for button_key, button in self.buttons.items():
                            if button['rect'].collidepoint(event.pos):
                                self.update_button_active(button_key, True)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        for button_key in self.buttons:
                            self.update_button_active(button_key, False)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.selected_piece = None
                        self.valid_moves = []

                    elif event.key == pygame.K_SPACE:
                        import random
                        self.view_offset_x = random.randint(200, 1000)
                        self.view_offset_y = random.randint(200, 700)

                    elif event.key == pygame.K_m:
                        # 模拟点击模式切换按钮
                        self.handle_button_click('mode_toggle')

                    elif event.key == pygame.K_r:
                        self.reset_game()
                        self.show_message("游戏已重新开始")

                    elif event.key == pygame.K_c:
                        self.view_offset_x = self.SCREEN_WIDTH // 2
                        self.view_offset_y = self.SCREEN_HEIGHT // 2
                        self.show_message("视角已居中")

            self.draw_board()
            self.draw_ui()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    """主函数"""
    print("启动中国跳棋游戏...")
    print("游戏说明:")
    print("1. 玩家1(红色)从右侧三角形开始，目标是对面的左侧三角形")
    print("2. 玩家2(蓝色)从左侧三角形开始，目标是对面的右侧三角形")
    print("3. 可以单步移动或跳过棋子进行跳跃")
    print("4. 连续跳跃在一次移动中完成")
    print("\n游戏模式:")
    print("- 普通模式: 跳跃距离至少为2，所有中间格子必须有棋子")
    print("- 镜像模式: 跳跃距离必须为偶数，正中格子有棋子，其他中间格子为空")
    print("\n控制:")
    print("- 鼠标左键: 选择棋子和移动")
    print("- 点击按钮: 切换模式/重新开始/居中视角")
    print("- ESC: 取消选择")
    print("- 空格: 随机视角")
    print("- M: 切换模式 (快捷键)")
    print("- R: 重新开始游戏 (快捷键)")
    print("- C: 居中视角 (快捷键)")
    print()

    # 创建并运行游戏
    game = ChineseCheckersGUI()
    game.run()


if __name__ == "__main__":
    main()