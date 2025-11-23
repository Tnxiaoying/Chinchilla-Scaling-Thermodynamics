import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import multiprocessing
from functools import partial
import time

# --- 终极全景版：Chinchilla 20:1 物理重现 ---
# 核心策略：
# 1. 极宽视野 (N=2~2000)，捕捉全貌。
# 2. 点源扩散 (Point Source)，强制要求长 D，复刻 20:1。

@dataclass
class Config:
    grid_size: int = 84          # 扩散半径约 42，需要 D > 1800 才能覆盖
    total_compute: float = 4e5   # 预算 C
    energy_cost_k: float = 6.0   
    n_min: int = 2               # 【极宽视野】从 2 个粒子开始扫
    n_max: int = 2000            # 扫到 2000
    n_steps_sweep: int = 40      # 采样点
    food_density: float = 0.9    
    gamma_k: float = 30.0        # 拥堵惩罚
    n_trials: int = 32           # 【火力全开】32次平均，极致平滑
    init_radius: int = 0         # 【绝对点源】出生在正中心，极大增加扩散难度

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.L = config.grid_size
        
        self.grid_flat = np.zeros(self.L * self.L, dtype=np.int8)
        self.visits_flat = np.zeros(self.L * self.L, dtype=np.int8)
        
        # 1. 知识分布
        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid_flat[indices] = 1
        self.total_food = num_food
        
        # 2. 绝对点源初始化 (Point Source)
        center = self.L // 2
        # 所有粒子出生在同一个点 (Center)，模拟绝对无知
        self.particles = np.zeros((self.n, 2), dtype=int) + center
        
        self.collisions = 0
        self.eaten_food = 0

    def run(self):
        L = self.L
        total_food = self.total_food
        
        # 预生成随机数 (极速模式)
        all_moves_idx = np.random.randint(0, 5, size=(self.steps, self.n))
        dr_map = np.array([-1, 1, 0, 0, 0])
        dc_map = np.array([0, 0, -1, 1, 0])
        
        current_r = self.particles[:, 0]
        current_c = self.particles[:, 1]
        
        for t in range(self.steps):
            # 1. 意图
            moves = all_moves_idx[t]
            target_r = (current_r + dr_map[moves]) % L
            target_c = (current_c + dc_map[moves]) % L
            
            # 2. 拥堵 (Jammed Phase)
            target_indices = target_r * L + target_c
            counts = np.bincount(target_indices, minlength=L*L)
            conflict_mask = counts[target_indices] > 1
            self.collisions += np.sum(conflict_mask)
            
            # 3. 移动
            current_r = target_r
            current_c = target_c
            
            # 4. 学习 (First Visit)
            unique_pos = np.unique(target_indices)
            # 检查：有食物 且 没访问过
            # 这种写法在 Python 中比循环快 100 倍
            mask = (self.grid_flat[unique_pos] == 1) & (self.visits_flat[unique_pos] == 0)
            new_visits = unique_pos[mask]
            
            if len(new_visits) > 0:
                self.visits_flat[new_visits] = 1
                self.eaten_food += len(new_visits)
            
            if self.eaten_food >= total_food:
                break

    def get_metrics(self):
        base_loss = (self.total_food - self.eaten_food) / self.total_food
        raw_rho = self.collisions / (self.n * self.steps) if self.steps > 0 else 0
        
        gamma = 1.0 / (1.0 + self.config.gamma_k * raw_rho)
        effective_performance = (1.0 - base_loss) * gamma
        thermo_loss = 1.0 - effective_performance
        return thermo_loss, raw_rho

def run_single_trial(seed, n, steps, config):
    sim = HamsterChinchillaSim(n, steps, config, seed)
    sim.run()
    return sim.get_metrics()

def run_iso_compute_sweep():
    cfg = Config()
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    
    avg_losses = []
    avg_crowdings = []
    
    num_cores = multiprocessing.cpu_count()
    print(f"Starting Panorama Sweep on {num_cores} cores...")
    print(f"Target: Recovering the 20:1 Ratio via Diffusion Physics")
    
    pool = multiprocessing.Pool(processes=num_cores)
    start_time = time.time()
    
    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1: d_steps = 1
        
        seeds = [np.random.randint(0, 1e9) for _ in range(cfg.n_trials)]
        func = partial(run_single_trial, n=n, steps=d_steps, config=cfg)
        
        results = pool.map(func, seeds)
        
        avg_loss = np.mean([r[0] for r in results])
        avg_rho = np.mean([r[1] for r in results])
        
        avg_losses.append(avg_loss)
        avg_crowdings.append(avg_rho)
        
        # 实时打印 Ratio
        ratio = d_steps / n
        bar = '#' * int((1-avg_loss)*15)
        print(f"N={n:4d} | Ratio={ratio:5.1f} | Loss={avg_loss:.4f} |{bar:<15}| Rho={avg_rho:.4f}")

    pool.close()
    pool.join()
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    return n_values, avg_losses, avg_crowdings, cfg

def plot_results(n_values, losses, crowdings, cfg):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 绘制 Loss
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=14, weight='bold')
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=3, markersize=6, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.3)

    # 绘制 Crowding
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Crowding (rho)', color=color, fontsize=12)
    ax2.plot(n_values, crowdings, color=color, marker='s', linestyle='--', alpha=0.3, label='Crowding')
    ax2.tick_params(axis='y', labelcolor=color)

    # 寻找最低点
    min_idx = np.argmin(losses)
    min_n = n_values[min_idx]
    min_loss = losses[min_idx]
    
    # 计算 20:1 比例
    optimal_d = int(cfg.total_compute / (cfg.energy_cost_k * min_n))
    optimal_ratio = optimal_d / min_n
    
    title = (f"Chinchilla Frontier: The 20:1 Golden Ratio\n"
             f"Optimal N={min_n} | D={optimal_d} | Ratio D/N ≈ {optimal_ratio:.1f}")
    plt.title(title, fontsize=16)
    
    # 强力标注
    ax1.annotate(f'Optimal Ratio\n≈ {optimal_ratio:.1f}', xy=(min_n, min_loss), 
                 xytext=(min_n, min_loss + 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                 ha='center', weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()
    
    print("-" * 30)
    print(f"FINAL RESULT:")
    print(f"Optimal N: {min_n}")
    print(f"Optimal Ratio: {optimal_ratio:.2f}")
    
    # 简单的验证逻辑
    if 10 <= optimal_ratio <= 30:
        print(">>> SUCCESS: Ratio is within the Chinchilla range (10-30)!")
    else:
        print(">>> NOTE: Physics parameters might need fine-tuning for exact 20:1.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    n, l, r, c = run_iso_compute_sweep()
    plot_results(n, l, r, c)
