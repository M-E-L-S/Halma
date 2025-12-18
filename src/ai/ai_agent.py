# ai_agent.py
#from src.ai.seacrch import Search
from src.ai.tradition_serach import Search


class AIPlayer:
    """AI玩家代理"""

    def __init__(self, player, depth=3):
        """
        初始化AI玩家

        参数:
            player: 玩家编号 (1 或 -1)
            depth: 搜索深度
        """
        self.player = player
        self.search_ai = Search(player, depth)
        self.name = f"AI Player {player}"

    def get_move(self, game_state):
        """
        根据当前游戏状态获取AI的移动

        参数:
            game_state: 游戏状态字典（来自game_state.py）

        返回:
            最佳移动
        """
        print(f"\n[AI Player {self.player} 正在思考...]")

        # 获取最佳移动
        best_move = self.search_ai.make_move(game_state)

        print(f"[AI 评估了 {self.search_ai.nodes_evaluated} 个节点]")
        print(f"[AI 进行了 {self.search_ai.pruning_count} 次剪枝]")

        if best_move:
            print(f"[AI 选择移动: {best_move}]")
        else:
            print("[AI 没有找到合法移动]")

        return best_move

    def reset(self):
        """重置AI状态"""
        self.search_ai.nodes_evaluated = 0
        self.search_ai.pruning_count = 0