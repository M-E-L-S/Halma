"""
模仿国际象棋AI鳕鱼(StockFish)的NNUE神经网络训练方式。
用于在alpha-beta搜索中加速局面评估计算且增加评估深度。
由于中国跳棋复杂度远低于国象，训练数据生成和模型训练都相对简单。
训练模型存储在nnue_chinese.pt中，并已经整合进了Search类的评估函数中，可以与手写传统评估共同使用。

author:徐温洌。
"""


import torch
import torch.nn as nn
import torch.optim as optim
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


from src.core.board import ChineseCheckersBoard
from src.core.moves import ChineseCheckersMoves
#from src.ai.tradition_serach import Search
#from src.ai.evaluator import ChineseCheckersEvaluator


# 构建索引映射
example_board = ChineseCheckersBoard()
ALL_CELLS = example_board.get_all_cells()
CELL_INDEX = {
    cell: i
    for i, cell in enumerate(ALL_CELLS)
}
INPUT_SIZE = len(ALL_CELLS) * 2


# 局面编码，作为NNUE输入
def encode_board(board):
    x = torch.zeros(INPUT_SIZE)

    for coord, piece in board.cells.items():
        idx = CELL_INDEX[coord]

        if piece == 1:
            x[idx] = 1
        elif piece == -1:
            x[idx + len(ALL_CELLS)] = 1

    return x


# 随机局面生成，用于训练
def generate_random_position():

    board = ChineseCheckersBoard()
    # 10-200步随机走子。模仿StockFish的生成方式
    steps = random.randint(10, 200)
    player = 1

    for _ in range(steps):
        moves = ChineseCheckersMoves.generate_all_moves(board, player)

        if not moves:
            break
        move = random.choice(moves)
        board = ChineseCheckersMoves.apply_move(board, move)
        player *= -1

    return board


# NNUE
class NNUE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Dataset
THREADS = 16
#ai_agent = Search(player=-1, depth=6)


# def generate_one_position(_):
#
#     board = generate_random_position()
#     ai_agent.eva = ChineseCheckersEvaluator(board)
#     score = ai_agent._alpha_beta(board, 3, False, float("-inf"), float("inf"), player=1)
#     x = encode_board(board)
#     score = score / 1000
#
#     return x, score

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, size):
#         self.data = []
#         print(f"Generating {size} positions using {THREADS} threads")
#
#         with ProcessPoolExecutor(max_workers=THREADS) as executor:
#             futures = [
#                 executor.submit(generate_one_position, i)
#                 for i in range(size)
#             ]
#
#             for i, future in enumerate(as_completed(futures)):
#                 x, score = future.result()
#                 self.data.append((x, score))
#
#                 if i % 100 == 0:
#                     print(
#                         f"{i}/{size} "
#                         f"{i/size*100:.1f}%"
#                     )
#
#     def __len__(self):
#
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         x, y = self.data[idx]
#
#         return x, torch.tensor([y], dtype=torch.float32)


# train
# def train():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = NNUE().to(device)
#     dataset = Dataset(50000)
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=256,
#         shuffle=True
#     )
#
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=0.001
#     )
#
#     loss_fn = nn.MSELoss()
#
#     for epoch in range(30):
#         total = 0
#         print(f"\n===== EPOCH {epoch} =====")
#
#         for batch_idx, (x, y) in enumerate(loader):
#             x = x.to(device)
#             y = y.to(device)
#             pred = model(x)
#             loss = loss_fn(pred, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total += loss.item()
#
#             if batch_idx % 20 == 0:
#                 print(
#                     f"[TRAIN] epoch={epoch} batch={batch_idx}/{len(loader)} "
#                     f"loss={loss.item():.6f} "
#                     f"pred_mean={pred.mean().item():.3f} "
#                     f"target_mean={y.mean().item():.3f}"
#                 )
#
#         print(
#             f"[EPOCH DONE] {epoch} "
#             f"total_loss={total:.4f} "
#             f"avg_loss={total / len(loader):.6f}"
#         )
#
#         torch.save(model.state_dict(), "nnue_chinese.pt")
#         print("[SAVE] model saved")
#
#     print("\n===== TEST INFERENCE =====")
#     test_board = generate_random_position()
#     test_x = encode_board(test_board).to(device)
#     pred = model(test_x).item()
#     ai_agent.eva = ChineseCheckersEvaluator(test_board)
#     target = ai_agent._alpha_beta(test_board, 3, False, float("-inf"), float("inf"), player=1) / 1000
#     print(f"[TEST] predicted score={pred * 100:.2f} target score={target * 100:.2f}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    #train()