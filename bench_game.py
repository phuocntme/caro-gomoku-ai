import os
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Thiết lập hạt giống ngẫu nhiên để tái lập kết quả
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Tự động chọn thiết bị tính toán: ưu tiên CUDA (GPU RTX 4050), fallback về CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# Thiết lập thư mục lưu trữ checkpoint cục bộ
PROJECT_DIR = Path(os.getcwd()).resolve()
SAVE_DIR = PROJECT_DIR / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Thư mục dự án: {PROJECT_DIR}")
print(f"Thư mục lưu Checkpoint cục bộ: {SAVE_DIR}")
print(f"Thiết bị sử dụng: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Chế độ AMP (FP16): {'Bật' if USE_AMP else 'Tắt'}")
else:
    print("Đang chạy trên CPU (khuyến nghị cài PyTorch CUDA để khai thác GPU RTX 4050).")

# Chọn chế độ: 'smoke_test' (kiểm tra nhanh) hoặc 'training' (huấn luyện thật)
MODE = "training"

@dataclass
class Config:
    """Cấu hình toàn bộ tham số cho môi trường, mô hình và huấn luyện."""

    # Tham số bàn cờ
    board_size: int = 15       # Chuẩn thi đấu cờ Caro 15x15 (225 ô)
    win_length: int = 5        # Số quân liên tiếp để chiến thắng theo luật Việt Nam

    # Tham số MCTS & Heuristic
    simulations: int = 150     # Số lượt mô phỏng MCTS mỗi nước đi
    c_puct: float = 1.5        # Hệ số cân bằng khám phá/khai thác PUCT
    dirichlet_alpha: float = 0.05   # Nhiễu Dirichlet tối ưu cho bàn 15x15 (225 ô)
    dirichlet_epsilon: float = 0.25 # Tỉ lệ pha trộn nhiễu Dirichlet
    temperature_moves: int = 10     # Số nước đầu dùng nhiệt độ tau=1.0 để khám phá

    # Tham số mạng SE-ResNet Thuần Tích Chập
    channels: int = 64         # Số kênh tích chập
    residual_blocks: int = 4   # Số khối SE-ResNet

    # Tham số huấn luyện & Opponent Pool
    learning_rate: float = 1e-3     # Tốc độ học của AdamW
    weight_decay: float = 1e-4      # Hệ số suy giảm trọng số L2
    batch_size: int = 64            # Kích thước batch huấn luyện
    buffer_capacity: int = 30000    # Sức chứa Replay Buffer
    self_play_games: int = 5        # Số ván tự đấu sinh dữ liệu mỗi vòng
    train_steps: int = 25           # Số bước cập nhật gradient mỗi vòng
    num_iterations: int = 50        # Tổng số vòng lặp huấn luyện
    save_interval: int = 5          # Chu kỳ lưu checkpoint định kỳ
    pool_size: int = 8              # Số model lịch sử lưu trong Opponent Pool
    pool_play_ratio: float = 0.20   # Tỉ lệ đấu với đối thủ lịch sử (20% ván)


# Áp dụng cấu hình theo chế độ đã chọn
if MODE == "training":
    config = Config(
        board_size=15,
        win_length=5,
        simulations=150,
        dirichlet_alpha=0.05,
        temperature_moves=10,
        channels=64,
        residual_blocks=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        buffer_capacity=30000,
        self_play_games=5,
        train_steps=25,
        num_iterations=50,
        save_interval=5,
        pool_size=8,
        pool_play_ratio=0.20
    )
else:
    # Cấu hình kiểm tra nhanh (Smoke Test)
    config = Config(
        board_size=10,
        win_length=5,
        simulations=8,
        dirichlet_alpha=0.3,
        temperature_moves=4,
        channels=32,
        residual_blocks=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        buffer_capacity=2000,
        self_play_games=1,
        train_steps=2,
        num_iterations=2,
        save_interval=1,
        pool_size=4,
        pool_play_ratio=0.0
    )

print(f"Chế độ hiện tại: [{MODE.upper()}]")
print(f"Cấu hình: Bàn cờ {config.board_size}x{config.board_size}, ăn {config.win_length} quân thắng (Luật Việt Nam).")
print(f"MCTS Simulations: {config.simulations} | Dirichlet Alpha: {config.dirichlet_alpha}")
print(f"SE-ResNet: {config.residual_blocks} blocks x {config.channels} channels (Thuần tích chập)")
print(f"Dự kiến chạy: {config.num_iterations} vòng lặp, mỗi vòng {config.self_play_games} ván tự đấu.")

class GomokuEnv:
    """Môi trường bàn cờ Caro (Gomoku) theo luật Việt Nam và biểu diễn 4 kênh."""

    def __init__(self, config: Config):
        self.size = config.board_size
        self.win_length = config.win_length
        self.reset()

    def reset(self):
        """Khởi tạo lại ván cờ về trạng thái rỗng."""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1  # 1: Đen (đi trước), -1: Trắng (đi sau)
        self.last_move = None
        self.done = False
        self.winner = 0          # 1: Đen thắng, -1: Trắng thắng, 0: Hòa/Chưa xong
        return self.state()

    def state(self) -> np.ndarray:
        """Trả về tensor trạng thái 4 kênh:
        Kênh 0: Vị trí quân của người chơi hiện tại.
        Kênh 1: Vị trí quân của đối thủ.
        Kênh 2: Nước đi gần nhất (1 tại last_move, 0 ở các ô khác).
        Kênh 3: Lợi thế lượt đi (toàn 1 nếu là Đen/Player 1, toàn 0 nếu là Trắng/Player -1).
        """
        s = np.zeros((4, self.size, self.size), dtype=np.float32)
        s[0] = (self.board == self.current_player).astype(np.float32)
        s[1] = (self.board == -self.current_player).astype(np.float32)
        if self.last_move is not None:
            s[2, self.last_move[0], self.last_move[1]] = 1.0
        if self.current_player == 1:
            s[3].fill(1.0)
        return s

    def check_win(self, x: int, y: int, board: np.ndarray = None) -> bool:
        """Kiểm tra thắng theo Luật Caro Việt Nam.
        Chuỗi đúng win_length quân (5 quân) và không bị đối thủ chặn cả 2 đầu.
        Nếu >= 6 quân không tính thắng.
        """
        b = self.board if board is None else board
        player = b[x, y]
        if player == 0:
            return False

        opponent = -player

        for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1

            # Quét chiều dương
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.size and 0 <= ny < self.size and b[nx, ny] == player:
                count += 1
                nx += dx
                ny += dy
            head_blocked = (
                not (0 <= nx < self.size and 0 <= ny < self.size)
                or b[nx, ny] == opponent
            )

            # Quét chiều âm
            px, py = x - dx, y - dy
            while 0 <= px < self.size and 0 <= py < self.size and b[px, py] == player:
                count += 1
                px -= dx
                py -= dy
            tail_blocked = (
                not (0 <= px < self.size and 0 <= py < self.size)
                or b[px, py] == opponent
            )

            if count == self.win_length and not (head_blocked and tail_blocked):
                return True

        return False

    def is_open_four(self, x: int, y: int, player: int, board: np.ndarray = None) -> bool:
        """Kiểm tra xem đặt quân player tại (x, y) có tạo thành 4 quân mở 2 đầu (Live 4) không."""
        b = self.board if board is None else board
        for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1

            # Quét chiều dương
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.size and 0 <= ny < self.size and b[nx, ny] == player:
                count += 1
                nx += dx
                ny += dy
            head_open = (0 <= nx < self.size and 0 <= ny < self.size) and (b[nx, ny] == 0)

            # Quét chiều âm
            px, py = x - dx, y - dy
            while 0 <= px < self.size and 0 <= py < self.size and b[px, py] == player:
                count += 1
                px -= dx
                py -= dy
            tail_open = (0 <= px < self.size and 0 <= py < self.size) and (b[px, py] == 0)

            if count == self.win_length - 1 and head_open and tail_open:
                return True

        return False

    def legal_actions(self) -> np.ndarray:
        """Danh sách chỉ số các ô trống hợp lệ."""
        return np.flatnonzero(self.board.ravel() == 0)

    def get_tactical_bonuses(self, player: int, board: np.ndarray = None) -> np.ndarray:
        """Tính điểm thưởng chiến thuật (Soft Prior) cho tất cả các ô hợp lệ.
        - Thắng ngay (5 quân): +15.0
        - Chặn đối thủ thắng ngay: +12.0
        - Tạo 4 mở 2 đầu (Live 4): +10.0
        - Chặn đối thủ tạo 4 mở (chặn Live 3 / Jump 3): +8.0
        """
        b = self.board if board is None else board
        bonuses = np.zeros(self.size * self.size, dtype=np.float32)
        empty = np.flatnonzero(b.ravel() == 0)
        opponent = -player

        for action in empty:
            x, y = divmod(int(action), self.size)

            # 1. Ta thắng ngay
            b[x, y] = player
            if self.check_win(x, y, board=b):
                bonuses[action] = max(bonuses[action], 15.0)
            if bonuses[action] < 15.0 and self.is_open_four(x, y, player, board=b):
                bonuses[action] = max(bonuses[action], 10.0)
            b[x, y] = 0

            # 2. Đối thủ thắng ngay (phải chặn)
            b[x, y] = opponent
            if self.check_win(x, y, board=b):
                bonuses[action] = max(bonuses[action], 12.0)
            if bonuses[action] < 10.0 and self.is_open_four(x, y, opponent, board=b):
                bonuses[action] = max(bonuses[action], 8.0)
            b[x, y] = 0

        return bonuses

    def tactical_action(self, current_player: int, board: np.ndarray = None):
        """Lấy nước đi chiến thuật khẩn cấp nhất (dùng cho Interactive Play / Hard check)."""
        bonuses = self.get_tactical_bonuses(current_player, board=board)
        best_act = np.argmax(bonuses)
        if bonuses[best_act] >= 8.0:
            return int(best_act)
        return None

    def step(self, action: int):
        """Thực hiện một nước đi trên bàn cờ."""
        if self.done:
            raise RuntimeError("Ván cờ đã kết thúc.")

        x, y = divmod(int(action), self.size)
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError(f"Tọa độ ({x}, {y}) nằm ngoài bàn cờ.")
        if self.board[x, y] != 0:
            raise ValueError(f"Ô ({x}, {y}) đã có quân cờ.")

        player = self.current_player
        self.board[x, y] = player
        self.last_move = (x, y)

        # Kiểm tra điều kiện kết thúc
        has_won = self.check_win(x, y)
        is_full = not np.any(self.board == 0)
        self.done = has_won or is_full
        self.winner = player if has_won else 0

        # Đổi lượt chơi
        self.current_player = -player
        return self.state(), self.winner, self.done

class SEBlock(nn.Module):
    """Khối chú ý kênh Squeeze-and-Excitation."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        scale = self.layers(self.pool(x).view(batch, channels))
        return x * scale.view(batch, channels, 1, 1)


class SEResBlock(nn.Module):
    """Khối Residual tích hợp SE Channel Attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + residual, inplace=True)


class GomokuNetAlphaZero(nn.Module):
    """Mạng Fully-Convolutional SE-ResNet Dual-Head chuẩn AlphaZero cho cờ Caro."""

    def __init__(self, config: Config):
        super().__init__()
        channels = config.channels

        # Lớp stem nhận tensor 4 kênh trạng thái
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Thân mạng gồm các khối SE-ResBlock
        self.backbone = nn.Sequential(*[
            SEResBlock(channels) for _ in range(config.residual_blocks)
        ])

        # Policy Head thuần tích chập (Fully-Convolutional)
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Flatten()  # (B, 1, H, W) -> (B, H * W)
        )

        # Value Head: Tích chập + Adaptive Pooling + MLP
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor):
        features = self.backbone(self.stem(state))
        policy_logits = self.policy(features)

        v = self.value_conv(features)
        v = F.adaptive_avg_pool2d(v, (1, 1)).flatten(1)
        value = self.value_fc(v).squeeze(-1)

        return policy_logits, value

class Node:
    """Nút trên cây tìm kiếm MCTS."""

    def __init__(self, prior: float = 0.0):
        self.prior = float(prior)
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
    """Bộ tìm kiếm Monte Carlo Tree Search theo chuẩn AlphaZero kết hợp Soft Prior & NumPy Vectorization."""

    def __init__(self, model: nn.Module, config: Config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    @torch.inference_mode()
    def evaluate(self, state: np.ndarray, legal_mask: np.ndarray, tactical_bonuses: np.ndarray = None):
        """Dùng mạng nơ-ron đánh giá prior xác suất và value của trạng thái kết hợp Soft Prior."""
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        logits, value = self.model(tensor)
        logits = logits[0].float().cpu().numpy()

        # Cộng điểm thưởng chiến thuật (Soft Prior) vào logits trước khi softmax
        if tactical_bonuses is not None:
            logits += tactical_bonuses

        # Che các nước đi không hợp lệ
        logits[~legal_mask] = -1e9
        probs = np.exp(logits - logits.max())
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs[legal_mask] = 1.0 / legal_mask.sum()
        return probs, float(value.item())

    def run(self, env: GomokuEnv, add_noise: bool = True, temperature: float = 1.0) -> np.ndarray:
        """Thực hiện chu kỳ mô phỏng MCTS và trả về vector xác suất nước đi."""
        root = Node()
        legal = env.board.ravel() == 0
        bonuses = env.get_tactical_bonuses(env.current_player)
        priors, _ = self.evaluate(env.state(), legal, tactical_bonuses=bonuses)
        actions = np.flatnonzero(legal)

        # Thêm nhiễu Dirichlet tại nút gốc khi tự đấu (đã tinh chỉnh theo bàn cờ)
        if add_noise and len(actions) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
            priors[actions] = (
                (1.0 - self.config.dirichlet_epsilon) * priors[actions]
                + self.config.dirichlet_epsilon * noise
            )

        for action in actions:
            root.children[int(action)] = Node(priors[action])

        # Vòng lặp mô phỏng
        for _ in range(self.config.simulations):
            board = env.board.copy()
            player = env.current_player
            last_mv = env.last_move
            node = root
            path = [node]

            # 1. Selection (Tối ưu hóa tốc độ bằng NumPy vectorization)
            while node.children:
                child_items = list(node.children.items())
                act_list = np.array([item[0] for item in child_items], dtype=np.int64)
                priors_arr = np.array([item[1].prior for item in child_items], dtype=np.float32)
                visits_arr = np.array([item[1].visit_count for item in child_items], dtype=np.float32)
                q_arr = np.array([item[1].q for item in child_items], dtype=np.float32)

                scale = math.sqrt(node.visit_count + 1)
                u = self.config.c_puct * priors_arr * scale / (1.0 + visits_arr)
                puct_scores = -q_arr + u
                best_idx = np.argmax(puct_scores)

                action = int(act_list[best_idx])
                node = child_items[best_idx][1]

                x, y = divmod(action, env.size)
                board[x, y] = player
                last_mv = (x, y)
                path.append(node)

                if env.check_win(x, y, board=board):
                    value = 1.0
                    break
                player = -player
            else:
                # 2. Expansion & Evaluation (dùng trạng thái 4 kênh)
                cur_state = np.zeros((4, env.size, env.size), dtype=np.float32)
                cur_state[0] = (board == player).astype(np.float32)
                cur_state[1] = (board == -player).astype(np.float32)
                if last_mv is not None:
                    cur_state[2, last_mv[0], last_mv[1]] = 1.0
                if player == 1:
                    cur_state[3].fill(1.0)

                cur_legal = board.ravel() == 0
                if not cur_legal.any():
                    value = 0.0  # Bàn cờ đầy: hòa
                else:
                    cur_bonuses = env.get_tactical_bonuses(player, board=board)
                    priors, value = self.evaluate(cur_state, cur_legal, tactical_bonuses=cur_bonuses)
                    for act in np.flatnonzero(cur_legal):
                        node.children[int(act)] = Node(priors[act])

            # 3. Backup (đổi dấu value khi truyền ngược)
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value
                value = -value

        # Tính phân phối xác suất từ số lượt thăm visit_count
        actions = np.array(list(root.children.keys()), dtype=np.int64)
        counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

        if temperature <= 1e-6 or counts.sum() == 0:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()

        policy = np.zeros(env.size * env.size, dtype=np.float32)
        policy[actions] = probs
        return policy

class ReplayBuffer:
    """Bộ đệm kinh nghiệm tái tạo ván đấu với D4 Augmentation."""

    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def add_game(self, records, winner: int):
        """Thêm các bước cờ từ ván đấu hoàn chỉnh vào buffer."""
        for state, policy, player in records:
            if winner == 0:
                target = 0.0
            elif winner == player:
                target = 1.0
            else:
                target = -1.0

            # Áp dụng 8 phép biến đổi đối xứng D4 trên tensor 4 kênh
            for aug_state, aug_policy in augment_d4(state, policy, state.shape[-1]):
                self.data.append((aug_state, aug_policy, target))

    def sample(self, batch_size: int):
        """Lấy ngẫu nhiên một mini-batch từ bộ đệm."""
        batch = random.sample(self.data, batch_size)
        states, policies, values = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(policies), dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.data)


class OpponentPool:
    """Bể đối thủ lịch sử (Opponent Pool) lưu trữ các phiên bản checkpoint cũ."""

    def __init__(self, max_size: int = 8):
        self.max_size = max_size
        self.pool = []

    def add(self, model: nn.Module):
        """Lưu một bản sao trọng số mô hình lên CPU."""
        state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)  # FIFO
        self.pool.append(state_dict)

    def sample_opponent(self, config: Config, device: torch.device):
        """Lấy ngẫu nhiên một mô hình lịch sử làm đối thủ thi đấu."""
        if not self.pool:
            return None
        state_dict = random.choice(self.pool)
        opponent = GomokuNetAlphaZero(config).to(device)
        opponent.load_state_dict(state_dict)
        opponent.eval()
        return opponent

    def __len__(self) -> int:
        return len(self.pool)

def self_play_game(model: nn.Module, config: Config, device: torch.device, opponent_model: nn.Module = None):
    """Tự đấu một ván hoàn chỉnh bằng MCTS để thu thập dữ liệu.
    Nếu có opponent_model: model chính đấu với opponent_model (ngẫu nhiên đổi lượt đi).
    """
    env = GomokuEnv(config)
    search_main = MCTS(model, config, device)
    search_opp = MCTS(opponent_model, config, device) if opponent_model is not None else search_main

    # 0 = self-play thuần túy; 1: model chính đi trước (Đen); -1: model chính đi sau (Trắng)
    main_player = random.choice([1, -1]) if opponent_model is not None else 0
    records = []

    while not env.done:
        state = env.state()
        cur_search = search_main if (main_player == 0 or env.current_player == main_player) else search_opp

        # Nhiệt độ cao ở các nước đầu để đa dạng hóa khai cuộc
        temperature = 1.0 if len(records) < config.temperature_moves else 0.1
        policy = cur_search.run(env, add_noise=True, temperature=temperature)

        # Chỉ lưu kinh nghiệm từ góc nhìn của model chính (hoặc cả 2 nếu self-play thuần)
        if main_player == 0 or env.current_player == main_player:
            records.append((state, policy, env.current_player))

        action = int(np.random.choice(len(policy), p=policy))
        env.step(action)

    return records, env.winner


def train_model(model: nn.Module, buffer: ReplayBuffer, config: Config,
                device: torch.device, optimizer: torch.optim.Optimizer,
                scaler: torch.amp.GradScaler):
    """Cập nhật trọng số mạng nơ-ron qua các mini-batch từ ReplayBuffer."""
    model.train()
    metrics = {"total": [], "policy": [], "value": []}

    for _ in range(config.train_steps):
        if len(buffer) < config.batch_size:
            break

        states, target_policy, target_value = buffer.sample(config.batch_size)
        states = states.to(device)
        target_policy = target_policy.to(device)
        target_value = target_value.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            logits, values = model(states)
            # Cross-Entropy Policy Loss
            policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(1).mean()
            # MSE Value Loss
            value_loss = F.mse_loss(values, target_value)
            loss = policy_loss + value_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics["total"].append(loss.item())
        metrics["policy"].append(policy_loss.item())
        metrics["value"].append(value_loss.item())

    return metrics


# Khởi tạo mô hình V2, tối ưu hóa, bộ đệm và đối thủ lịch sử
model = GomokuNetAlphaZero(config).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)
scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)
replay_buffer = ReplayBuffer(config.buffer_capacity)
opponent_pool = OpponentPool(config.pool_size)

# Đường dẫn checkpoint V2 riêng biệt để không xung đột phiên bản cũ
latest_ckpt_path = os.path.join(SAVE_DIR, f"caro_v2_model_{config.board_size}x{config.board_size}_latest.pth")
best_ckpt_path = os.path.join(SAVE_DIR, f"caro_v2_model_{config.board_size}x{config.board_size}_best.pth")
history_log_path = os.path.join(SAVE_DIR, f"caro_v2_history_{config.board_size}x{config.board_size}.json")

history = {"total": [], "policy": [], "value": []}
start_iteration = 0
best_loss = float("inf")

# Phục hồi checkpoint V2 nếu có sẵn
if os.path.exists(latest_ckpt_path):
    try:
        ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Đã nạp lại checkpoint V2 từ: {latest_ckpt_path} (Tiếp tục từ vòng {start_iteration + 1})")
        if os.path.exists(history_log_path):
            with open(history_log_path, "r", encoding="utf-8") as hf:
                history = json.load(hf)
            print(f"Đã nạp lịch sử huấn luyện: {len(history['total'])} vòng trước đó.")
    except Exception as e:
        print(f"Không thể nạp checkpoint V2 cũ ({e}). Khởi tạo huấn luyện V2 từ đầu.")
else:
    print("Khởi tạo mô hình V2 mới từ đầu (chưa có checkpoint V2).")

# Đưa bản khởi tạo đầu tiên vào Opponent Pool
opponent_pool.add(model)

# Vòng lặp huấn luyện chính
total_iterations = config.num_iterations
print(f"\nBắt đầu huấn luyện V2: từ vòng {start_iteration + 1} đến {start_iteration + total_iterations}...\n")

for it in tqdm(range(start_iteration, start_iteration + total_iterations), desc="Tiến trình tự đấu & Train"):
    model.eval()
    for _ in range(config.self_play_games):
        # Quyết định đấu với chính mình hay đấu với đối thủ trong Opponent Pool
        use_pool = (len(opponent_pool) > 0) and (random.random() < config.pool_play_ratio)
        opp_net = opponent_pool.sample_opponent(config, DEVICE) if use_pool else None

        game_records, game_winner = self_play_game(model, config, DEVICE, opponent_model=opp_net)
        replay_buffer.add_game(game_records, game_winner)

    metrics = train_model(model, replay_buffer, config, DEVICE, optimizer, scaler)
    for key in history:
        if metrics[key]:
            history[key].append(float(np.mean(metrics[key])))

    cur_total_loss = history["total"][-1] if history["total"] else 0.0
    cur_policy_loss = history["policy"][-1] if history["policy"] else 0.0
    cur_value_loss = history["value"][-1] if history["value"] else 0.0

    print(f"Vòng {it + 1}: "
          f"Tổng loss = {cur_total_loss:.4f}, "
          f"Policy loss = {cur_policy_loss:.4f}, "
          f"Value loss = {cur_value_loss:.4f} "
          f"(Bộ đệm: {len(replay_buffer)} mẫu | Pool: {len(opponent_pool)} models)")

    # Lưu checkpoint định kỳ hoặc bản tốt nhất
    ckpt_data = {
        "iteration": it + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "config": config.__dict__
    }
    torch.save(ckpt_data, latest_ckpt_path)

    if cur_total_loss > 0 and cur_total_loss < best_loss:
        best_loss = cur_total_loss
        ckpt_data["best_loss"] = best_loss
        torch.save(ckpt_data, best_ckpt_path)
        print(f"  -> Đã cập nhật best checkpoint mới: Loss = {best_loss:.4f}")

    if (it + 1) % config.save_interval == 0:
        periodic_path = os.path.join(SAVE_DIR, f"caro_v2_model_{config.board_size}x{config.board_size}_iter_{it + 1}.pth")
        torch.save(ckpt_data, periodic_path)
        opponent_pool.add(model)  # Cập nhật model vào Opponent Pool

    with open(history_log_path, "w", encoding="utf-8") as hf:
        json.dump(history, hf, indent=2)

print(f"\nHuấn luyện V2 hoàn thành! Toàn bộ checkpoint đã lưu an toàn tại: {SAVE_DIR}")

print(f'Starting 1 self_play_game on {DEVICE} with 15x15, {config.simulations} sims...')
t0 = time.time()
records, winner = self_play_game(model, config, DEVICE)
t1 = time.time()
print(f'1 self-play game completed in {t1 - t0:.1f}s! Total moves: {len(records)}, Winner: {winner}')
