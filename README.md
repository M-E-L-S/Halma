# 中国跳棋智能体
### 一个基于Python实现的中国跳棋智能体，初版使用对抗搜索进行决策。

## 项目概述
项目初步使用手写启发式评估函数，先测试项目可行性

## 环境要求
- Python 3.8+
- Pygame（用于图形界面）
- Pytorch（用于深度学习扩展）

## 项目结构
```text
Halma/
├── src/                    # 源代码目录
│   ├── core/              # 核心逻辑
│   │   ├── game.py        # 游戏控制
│   │   └──...
│   ├── ai/                # AI算法
│   │   ├── search.py      # 搜索算法（Alpha-Beta剪枝）
│   │   ├── evaluator.py   # 评估函数
│   │   └── agent.py       # 智能体接口
│   └── utils/             # 工具函数
│       └──...
├── requirements.txt       # Python依赖
├── .gitignore             # Git忽略文件
└── README.md             # 项目说明
```


## 开发指南
- 添加新的评估函数
    - 在 src/ai/evaluator.py 中创建新的评估类 
    - 实现 evaluate() 方法

- 修改搜索算法
  - 编辑 src/ai/search.py 中的 alpha_beta_search 函数