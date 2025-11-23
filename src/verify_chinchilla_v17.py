import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


# --- v17: warmup + rho-threshold version ---
# 目标：
# 1）保留 v16 的点源初始化、等算力 C = 6 N D 结构；
# 2）给系统一个“热身期”，避免出生瞬间的必然拥堵直接拉高 rho；
# 3）对 rho 引入一个阈值 rho_threshold，只惩罚真正的长期拥堵；
# 这样更有利于显现真正的三相结构和 U 型曲线。


@dataclass
class Config:
    # 基本几何与算力约束
    grid_size: int = 80          # 地图边长 L
    total_compute: float = 4e4   # 总算力 C
    energy_cost_k: float = 6.0   # 能耗系数 k

    # N 扫描范围
    n_min: int = 2
    n_max: int = 150
    n_steps_sweep: int = 35

    # 环境 & 拥堵相关参数
    food_density: float = 0.9
    gamma_k: float = 10.0        # 拥堵惩罚强度（比 v16 柔和）
    rho_threshold: float = 0.02  # 拥堵阈值，小于该值视为“可接受”

    # 动力学与采样
    n_trials: int = 50
    init_radius: int = 0         # 仍然采用点源初始化
    sensor_noise: float = 0.15
    actuator_noise: float = 0.05

    # 新增：热身期比例（前 warmup_frac 的 step 不计入拥堵）
    warmup_frac: float = 0.1     # 例如 steps 的前 10% 视为“扩散期”


class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config: Config):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.L = config.grid_size

        # 网格：1 表示有食物
        self.grid_flat = np.zeros(self.L * self.L, dtype=np.uint8)
        self.visits_flat = np.zeros(self.L * self.L, dtype=np.uint8)

        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid_flat[indices] = 1
        self.total_food = num_food

        # 点源初始化：所有粒子从中心出发
        center = self.L // 2
        self.particles = np.zeros((self.n, 2), dtype=int) + center

        # 拥堵统计
        self.collisions = 0
        self.collision_steps = 0   # 参与 rho 统计的步数
        self.eaten_food = 0

        # 实际执行步数（遇到全吃完时可能提前结束）
        self.steps_executed = 0

    def run(self):
        L = self.L
        total_food = self.total_food

        # 预采样移动方向：0,1,2,3,4 -> 上下左右停
        all_moves_idx = np.random.randint(0, 5, size=(self.steps, self.n), dtype=np.int8)
        dr_map = np.array([-1, 1, 0, 0, 0], dtype=np.int8)
        dc_map = np.array([0, 0, -1, 1, 0], dtype=np.int8)

        sensor_noise_p = self.config.sensor_noise
        slip_p = self.config.actuator_noise
        rand_floats = np.random.rand(self.steps, self.n * 2).astype(np.float32)

        current_r = self.particles[:, 0]
        current_c = self.particles[:, 1]

        warmup_steps = int(self.steps * self.config.warmup_frac)

        for t in range(self.steps):
            moves = all_moves_idx[t]
            dr = dr_map[moves]
            dc = dc_map[moves]

            # 执行器噪声：一部分粒子“打滑”停在原地
            is_slip = rand_floats[t, :self.n] < slip_p
            dr[is_slip] = 0
            dc[is_slip] = 0

            target_r = (current_r + dr) % L
            target_c = (current_c + dc) % L

            # 目标格索引（用于统计访问与拥堵）
            target_indices = target_r * L + target_c

            # 仅在热身期之后统计拥堵
            if t >= warmup_steps:
                counts = np.bincount(target_indices, minlength=L * L)
                conflict_mask = counts[target_indices] > 1
                self.collisions += np.sum(conflict_mask)
                self.collision_steps += 1

            # 更新位置
            current_r = target_r
            current_c = target_c

            # 新访问的格子里若有食物则“吃掉”
            unique_pos = np.unique(target_indices)
            potential_new = unique_pos[
                (self.grid_flat[unique_pos] == 1) &
                (self.visits_flat[unique_pos] == 0)
            ]

            if len(potential_new) > 0:
                if sensor_noise_p > 0:
                    success_mask = np.random.rand(len(potential_new)) > sensor_noise_p
                    real_new = potential_new[success_mask]
                else:
                    real_new = potential_new

                if len(real_new) > 0:
                    self.visits_flat[real_new] = 1
                    self.eaten_food += len(real_new)

            self.steps_executed = t + 1

            if self.eaten_food >= total_food:
                break

    def get_metrics(self):
        # 基础 Loss：没被访问（吃到）的信息占比
        base_loss = (self.total_food - self.eaten_food) / self.total_food

        # 拥堵统计只基于热身之后的步数
        effective_steps = max(self.collision_steps, 1)
        raw_rho = self.collisions / (self.n * effective_steps)

        # 减去一个“安全阈值”，只惩罚超出部分
        adjusted_rho = max(0.0, raw_rho - self.config.rho_threshold)

        # 将拥堵转化为热效率因子 gamma
        gamma = 1.0 / (1.0 + self.config.gamma_k * adjusted_rho)

        effective_performance = (1.0 - base_loss) * gamma
        thermo_loss = 1.0 - effective_performance
        return thermo_loss, raw_rho


def run_iso_compute_sweep():
    cfg = Config()
    n_values = np.logspace(
        np.log10(cfg.n_min),
        np.log10(cfg.n_max),
        cfg.n_steps_sweep
    ).astype(int)
    n_values = np.unique(n_values)

    avg_losses = []
    avg_crowdings = []

    print(f"Starting v17 Warmup+Threshold Sweep (N={cfg.n_min}..{cfg.n_max})...")
    start_time = time.time()

    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1:
            d_steps = 1

        losses = []
        rhos = []

        for _ in range(cfg.n_trials):
            sim = HamsterChinchillaSim(n, d_steps, cfg)
            sim.run()
            l, r = sim.get_metrics()
            losses.append(l)
            rhos.append(r)

        avg_loss = np.mean(losses)
        avg_rho = np.mean(rhos)

        avg_losses.append(avg_loss)
        avg_crowdings.append(avg_rho)

        ratio = d_steps / n
        bar = '#' * int((1 - avg_loss) * 20)
        print(f"N={n:3d} | Ratio={ratio:7.1f} | Loss={avg_loss:.4f} |{bar:<20}|")

    print(f"Total time: {time.time() - start_time:.2f}s")
    return n_values, avg_losses, avg_crowdings, cfg


def plot_results(n_values, losses, crowdings, cfg: Config):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)')
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=12)
    ax1.semilogx(n_values, losses, color=color, marker='o', label='Thermodynamic Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", alpha=0.4)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Crowding (rho)', color=color, fontsize=12)
    ax2.plot(n_values, crowdings, color=color, marker='s',
             linestyle='--', alpha=0.3, label='Crowding')
    ax2.tick_params(axis='y', labelcolor=color)

    # 找 Loss 最低点
    losses_arr = np.array(losses)
    min_idx = np.argmin(losses_arr)
    min_n = int(n_values[min_idx])
    min_loss = float(losses_arr[min_idx])

    optimal_d = int(cfg.total_compute / (cfg.energy_cost_k * min_n))
    optimal_ratio = optimal_d / min_n

    title = (f"Chinchilla Frontier (v17): U-Shape with Warmup\\n"
             f"Optimal N={min_n} | Ratio D/N ≈ {optimal_ratio:.1f}")
    plt.title(title, fontsize=16)

    ax1.annotate(
        f'Optimal N={min_n}\\nRatio ≈ {optimal_ratio:.1f}',
        xy=(min_n, min_loss),
        xytext=(min_n, min_loss + 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=3),
        ha='center',
        weight='bold',
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_values, losses, rhos, cfg = run_iso_compute_sweep()
    plot_results(n_values, losses, rhos, cfg)

