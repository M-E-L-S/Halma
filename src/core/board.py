class CubeCoord:
    def __init__(self, q, r, s=None):
        if s is None:
            s = -q - r
        assert q + r + s == 0, f"Invalid cube coordinates: ({q},{r},{s}) sum={q + r + s}"
        self.q = q
        self.r = r
        self.s = s
    def __add__(self, other):
        return CubeCoord(self.q + other.q, self.r + other.r, self.s + other.s)
    def __sub__(self, other):
        return CubeCoord(self.q - other.q, self.r - other.r, self.s - other.s)
    def __mul__(self, scalar):
        return CubeCoord(self.q * scalar, self.r * scalar, self.s * scalar)
    def __floordiv__(self, other):
        """本身除以other"""
        return CubeCoord(self.q // other, self.r // other, self.s // other)
    def __eq__(self, other):
        return self.q == other.q and self.r == other.r and self.s == other.s
    def __neg__(self):
        return CubeCoord(-self.q, -self.r, -self.s)
    def __hash__(self):
        return hash((self.q, self.r, self.s))
    def __repr__(self):
        return f"({self.q},{self.r},{self.s})"
    def distance(self, other):
        vec = self - other
        return max(abs(vec.q), abs(vec.r), abs(vec.s))
    def neighbor(self, direction):
        return self + CubeCoord.hex_directions[direction % 6]
    def direction_to(self, other):
        # 获取到另一个格子的方向向量（如果是直线）
        if self == other:
            return CubeCoord(0, 0, 0)
        vec = other - self
        # 找到最大公约数
        from math import gcd
        # 检查是否共线
        if vec.q == 0 and vec.r == 0 and vec.s == 0:
            return CubeCoord(0, 0, 0)
        # 找到非零分量
        non_zero = [abs(x) for x in [vec.q, vec.r, vec.s] if x != 0]
        if not non_zero:
            return CubeCoord(0, 0, 0)
        # 计算最大公约数
        current_gcd = non_zero[0]
        for num in non_zero[1:]:
            current_gcd = gcd(current_gcd, num)
        # 归一化向量
        unit_vec = CubeCoord(vec.q // current_gcd, vec.r // current_gcd, vec.s // current_gcd)
        # 检查是否是基本方向
        for dir_vec in CubeCoord.hex_directions:
            if unit_vec == dir_vec:
                return unit_vec
        # 如果不是基本方向，返回None（不是直线）
        return None
    def is_straight_line_to(self, other):
        """检查是否在一条直线上"""
        return self.direction_to(other) is not None
    def cells_between(self, other):
        """获取两个格子之间的所有格子（不包括起点和终点）"""
        direction = self.direction_to(other)
        if direction is None:
            return []
        distance = self.distance(other)
        cells = []
        for step in range(1, distance):
            cell = self + (direction * step)
            cells.append(cell)
        return cells

    def is_neighbor(self, other):
        return self.distance(other) == 1
# 立方坐标
CubeCoord.hex_directions = [
    CubeCoord(1, 0, -1),  # 0: 东
    CubeCoord(1, -1, 0),  # 1: 东北
    CubeCoord(0, -1, 1),  # 2: 西北
    CubeCoord(-1, 0, 1),  # 3: 西
    CubeCoord(-1, 1, 0),  # 4: 西南
    CubeCoord(0, 1, -1)  # 5: 东南
]
class ChineseCheckersBoard:
    """中国跳棋棋盘：六边形+三角形"""

    def __init__(self):
        self.hex_radius = 4  # 中心六边形半径
        self.triangle_size = 4  # 三角形边长
        self.special_mode = False  # 当前模式为普通模式

        self.cells = {}  # 0: 空, 1: 玩家1, -1: 玩家2
        self.regions = {}


        self.player_start_regions = {
            1: 'tri1',  # 玩家1起始区域
            -1: 'tri4'  # 玩家2起始区域
        }
        self.player_target_regions = {
            1: 'tri4',  # 玩家1目标区域
            -1: 'tri1'  # 玩家2目标区域
        }

        self._init_board()

    def _init_board(self):
        """初始化棋盘"""
        self.cells.clear()
        self.regions.clear()

        self._create_hexagon()

        self._create_all_triangles()

        self._setup_chinese_checkers_pieces()

    def _create_all_triangles(self):
        """创建6个三角形区域"""
        for edge in range(6):
            self._create_edge_triangle(edge)

    def _create_edge_triangle(self, edge_index):
        """沿指定边创建三角形区域"""
        outward_dir = self._get_edge_outward_dir(edge_index)
        dir1_idx = edge_index
        dir2_idx = (edge_index + 1) % 6
        dir1 = CubeCoord.hex_directions[dir1_idx]
        dir2 = CubeCoord.hex_directions[dir2_idx]
        edge_start = CubeCoord(0, 0, 0)
        for _ in range(self.hex_radius):
            edge_start = edge_start + dir1
        triangle_cells = []
        for layer in range(self.triangle_size):
            width = 5 - layer
            point = edge_start
            for step in range(1,width):
                point = point + dir2
                triangle_cells.append(point)
            edge_start = edge_start + (dir2 - dir1)

        for coord in triangle_cells:
            if coord not in self.cells:
                self.cells[coord] = 0
                self.regions[coord] = f'tri{edge_index}'

    def _get_edge_outward_dir(self, edge_index):
        """获取从边向外延伸的方向"""

        dir1_idx = edge_index
        dir2_idx = (edge_index + 1) % 6
        dir1 = CubeCoord.hex_directions[dir1_idx]
        dir2 = CubeCoord.hex_directions[dir2_idx]
        # 边方向 = dir2 - dir1
        edge_dir = dir2 - dir1
        return -(dir1 + dir2)


    def _create_hexagon(self):
        """创建中心六边形区域"""
        radius = self.hex_radius
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                coord = CubeCoord(q, r)
                self.cells[coord] = 0  # 空
                self.regions[coord] = 'hex'

    def _setup_chinese_checkers_pieces(self):
        """设置中国跳棋初始棋子"""
        for coord in self.cells:
            self.cells[coord] = 0

        for player in [1, -1]:
            start_region = self.player_start_regions[player]

            region_cells = [c for c, r in self.regions.items() if r == start_region]

            # 按照到中心(0,0,0)的距离排序
            region_cells.sort(key=lambda c: c.distance(CubeCoord(0, 0, 0)))

            for coord in region_cells[:10]:
                self.cells[coord] = player

    def is_valid_cell(self, coord):
        return coord in self.cells

    def get_piece(self, coord):
        return self.cells.get(coord, 0)

    def set_piece(self, coord, piece):
        if self.is_valid_cell(coord):
            self.cells[coord] = piece

    def is_empty(self, coord):
        return self.get_piece(coord) == 0

    def has_piece(self, coord):
        return self.get_piece(coord) != 0

    def get_region(self, coord):
        return self.regions.get(coord)

    def get_all_cells(self):
        return list(self.cells.keys())

    def get_current_mode(self):
        return self.special_mode

    def change_mode(self):
        self.special_mode = not self.special_mode

    def get_player_pieces(self, player):
        """获取玩家的所有棋子坐标"""
        return [coord for coord, piece in self.cells.items()
                if piece == player]

    def copy(self):
        """深拷贝棋盘"""
        import copy
        new_board = ChineseCheckersBoard.__new__(ChineseCheckersBoard)
        new_board.hex_radius = self.hex_radius
        new_board.triangle_size = self.triangle_size
        new_board.cells = copy.deepcopy(self.cells)
        new_board.regions = copy.deepcopy(self.regions)
        new_board.player_start_regions = self.player_start_regions.copy()
        new_board.player_target_regions = self.player_target_regions.copy()
        new_board.special_mode = self.special_mode
        return new_board

    def print_board_info(self):
        """打印棋盘信息"""
        print("\n=== 中国跳棋棋盘 ===")
        print(f"总格子数: {len(self.cells)}")

        # 棋子统计
        player1_pieces = len([p for p in self.cells.values() if p == 1])
        player2_pieces = len([p for p in self.cells.values() if p == -1])

        print(f"玩家1棋子数: {player1_pieces} (起始区域: tri0, 目标区域: tri3)")
        print(f"玩家2棋子数: {player2_pieces} (起始区域: tri3, 目标区域: tri0)")

        # 显示目标区域状态
        for player in [1, -1]:
            target_region = self.player_target_regions[player]
            pieces_in_target = len([
                coord for coord in self.get_player_pieces(player)
                if self.regions[coord] == target_region
            ])
            print(f"玩家{player}在目标区域{target_region}: {pieces_in_target}/10")
