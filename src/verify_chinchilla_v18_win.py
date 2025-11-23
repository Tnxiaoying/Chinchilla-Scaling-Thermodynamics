import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


# ----------------------------
# v18: Capacity + Jam 模型
# ----------------------------
# 在 v17 的基础上增加：
# 1）每个粒子有“记忆上限” memory_per_particle（最多能学多少格子）；
# 2）吃东西时按粒子逐个检查，超过记忆上限就不再计入 eaten_food；
# 这样：
#   - N 太小：总容量 = N * memory_per_particle < total_food，左侧 Loss 偏高；
#   - N 适中：容量够用，同时还没严重拥堵 → Loss 最低；
#   - N 太大：步数太短 + 拥堵 γ 降低效率 → Loss 再次升高。


@dataclass
class Config:
    # 几何 & 算力约束
    grid_size: int = 80              # 地图边长 L
    total_compute: float = 4e4       # 总算力 C
    energy_cost_k: float = 6.0       # 单步能耗 k

    # N 扫描范围
    n_min: int = 2
    n_max: int = 150
    n_steps_sweep: int = 35

    # 环境 & 拥堵
    food_density: float = 0.9
    gamma_k: float = 10.0            # 拥堵惩罚强度
    rho_threshold: float = 0.02      # 拥堵阈值，小于该值视为“可接受”

    # 采样
    n_trials: int = 30               # 每个 N 平均次数（可按需调大）
    init_radius: int = 0             # 点源初始化
    sensor_noise: float = 0.15
    actuator_noise: float = 0.05

    # 热身期：前 warmup_frac 部分步骤不计入拥堵统计
    warmup_frac: float = 0.1

    # 新增：每个粒子的“记忆上限”（最多能学多少格子）
    memory_per_particle: int = 80    # 可按需要调，比如 60~120 区间试


class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config: Config):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.L = config.grid_size

        # 网格：1 = 有食物
        self.grid_flat = np.zeros(self.L * self.L, dtype=np.uint8)
        self.visits_flat = np.zeros(self.L * self.L, dtype=np.uint8)

        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid_flat[indices] = 1
        self.total_food = num_food

        # 粒子初始位置：全部在中心
        center = self.L // 2
        self.particles = np.zeros((self.n, 2), dtype=int) + center

        # 拥堵 & 学习统计
        self.collisions = 0
        self.collision_steps = 0
        self.eaten_food = 0
        self.steps_executed = 0

        # 每个粒子的已学习格子数
        self.memory_used = np.zeros(self.n, dtype=np.int32)

    def run(self):
        L = self.L
        total_food = self.total_food
        max_memory = self.config.memory_per_particle

        # 预采样移动方向
        all_moves_idx = np.random.randint(0, 5, size=(self.steps, self.n), dtype=np.int8)
        dr_map = np.array([-1, 1, 0, 0, 0], dtype=np.int8)
        dc_map = np.array([0, 0, -1, 1, 0], dtype=np.int8)

        sensor_noise_p = self.config.sensor_noise
        slip_p = self.config.actuator_noise
        # 每步给每个粒子准备 2 个随机数：一个给 slip，一个给 sensor noise
        rand_floats = np.random.rand(self.steps, self.n * 2).astype(np.float32)

        current_r = self.particles[:, 0]
        current_c = self.particles[:, 1]

        warmup_steps = int(self.steps * self.config.warmup_frac)

        for t in range(self.steps):
            moves = all_moves_idx[t]
            dr = dr_map[moves]
            dc = dc_map[moves]

            # 执行器噪声：部分粒子打滑停在原地
            is_slip = rand_floats[t, :self.n] < slip_p
            dr[is_slip] = 0
            dc[is_slip] = 0

            target_r = (current_r + dr) % L
            target_c = (current_c + dc) % L
            target_indices = target_r * L + target_c

            # 统计拥堵：只在热身期之后
            if t >= warmup_steps:
                counts = np.bincount(target_indices, minlength=L * L)
                conflict_mask = counts[target_indices] > 1
                self.collisions += int(np.sum(conflict_mask))
                self.collision_steps += 1

            # 学习 / 吃食物：带容量上限 + 传感噪声
            # 注意：逐粒子检查，确保同一格子只被第一个成功学习的粒子计一次
            for i in range(self.n):
                idx = target_indices[i]
                if self.grid_flat[idx] == 1 and self.visits_flat[idx] == 0:
                    if self.memory_used[i] < max_memory:
                        # 传感噪声：有一定概率“错过”这块信息
                        if sensor_noise_p <= 0 or rand_floats[t, self.n + i] > sensor_noise_p:
                            self.visits_flat[idx] = 1
                            self.eaten_food += 1
                            self.memory_used[i] += 1

            # 更新粒子位置
            current_r = target_r
            current_c = target_c

            self.steps_executed = t + 1
            if self.eaten_food >= total_food:
                break

    def get_metrics(self):
        # 基础损失：未学会的信息占比
        base_loss = (self.total_food - self.eaten_food) / self.total_food

        # 拥堵：按“后热身期”的平均冲突率
        effective_steps = max(self.collision_steps, 1)
        raw_rho = self.collisions / (self.n * effective_steps)

        # 只惩罚超过阈值的拥堵
        adjusted_rho = max(0.0, raw_rho - self.config.rho_threshold)

        # 拥堵 → 热效率 γ
        gamma = 1.0 / (1.0 + self.config.gamma_k * adjusted_rho)

        effective_performance = (1.0 - base_loss) * gamma
        thermo_loss = 1.0 - effective_performance
        return thermo_loss, raw_rho


def run_iso_compute_sweep():
    cfg = Config()

    # N 对数扫描
    n_values = np.logspace(
        np.log10(cfg.n_min),
        np.log10(cfg.n_max),
        cfg.n_steps_sweep
    ).astype(int)
    n_values = np.unique(n_values)

    avg_losses = []
    avg_crowdings = []

    print(f"Starting v18 Capacity+Jam Sweep (N={cfg.n_min}..{cfg.n_max})...")
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

        avg_loss = float(np.mean(losses))
        avg_rho = float(np.mean(rhos))

        avg_losses.append(avg_loss)
        avg_crowdings.append(avg_rho)

        ratio = d_steps / n
        bar = '#' * int((1 - avg_loss) * 20)
        print(f"N={n:3d} | Ratio={ratio:7.1f} | Loss={avg_loss:.4f} |{bar:<20}|")

    print(f"Total time: {time.time() - start_time:.2f}s")
    return n_values, avg_losses, avg_crowdings, cfg


def plot_results(n_values, losses, crowdings, cfg: Config):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 红色：热力学 Loss
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)')
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=12)
    ax1.semilogx(n_values, losses, color=color, marker='o', label='Thermodynamic Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", alpha=0.4)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)

    # 蓝色：拥堵 ρ
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

    title = (f"Chinchilla Frontier (v18): Capacity + Jam U-Shape\n"
             f"Optimal N={min_n} | Ratio D/N ≈ {optimal_ratio:.1f}")
    plt.title(title, fontsize=16)

    ax1.annotate(
        f'Optimal N={min_n}\nRatio ≈ {optimal_ratio:.1f}',
        xy=(min_n, min_loss),
        xytext=(min_n, min_loss + 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=3),
        ha='center',
        weight='bold',
        fontsize=11
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_values, losses, rhos, cfg = run_iso_compute_sweep()
    plot_results(n_values, losses, rhos, cfg)

