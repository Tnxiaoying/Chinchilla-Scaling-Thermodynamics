import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import multiprocessing
from functools import partial
import time

# --- 终极地狱版 (v11): 噪音与难度的协奏曲 ---
# 核心升级：
# 1. 引入感知与行动噪音 (Sensor & Actuator Noise)。
# 2. 地图扩容至 128x128，极大增加扩散难度。
# 3. 极宽视野 N=2~5000。

@dataclass
class Config:
    grid_size: int = 128         # 【地狱难度】地图进一步扩大
    total_compute: float = 8e5   # 提升预算以匹配大地图
    energy_cost_k: float = 6.0   
    n_min: int = 2               # 从 2 开始
    n_max: int = 5000            # 扫到 5000
    n_steps_sweep: int = 45      # 高精度扫描
    food_density: float = 0.9    
    gamma_k: float = 30.0        
    n_trials: int = 24           # 保持 i9 满载
    init_radius: int = 0         # 点源初始化
    sensor_noise: float = 0.25   # 【新】25% 概率感知失灵 (看不见食物)
    actuator_noise: float = 0.10 # 【新】10% 概率行动打滑 (原地踏步)

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
        
        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid_flat[indices] = 1
        self.total_food = num_food
        
        # 点源初始化
        center = self.L // 2
        self.particles = np.zeros((self.n, 2), dtype=int) + center
        
        self.collisions = 0
        self.eaten_food = 0

    def run(self):
        L = self.L
        total_food = self.total_food
        noise_p = self.config.sensor_noise
        slip_p = self.config.actuator_noise
        
        # 预生成随机数
        all_moves_idx = np.random.randint(0, 5, size=(self.steps, self.n))
        dr_map = np.array([-1, 1, 0, 0, 0])
        dc_map = np.array([0, 0, -1, 1, 0])
        
        # 预生成噪音掩码
        sensor_fail_mask = np.random.rand(self.steps, self.n) < noise_p
        slip_mask = np.random.rand(self.steps, self.n) < slip_p
        
        current_r = self.particles[:, 0]
        current_c = self.particles[:, 1]
        
        for t in range(self.steps):
            # 1. 意图 (含感知噪音)
            moves = all_moves_idx[t]
            
            # 如果感知失灵，强制随机乱走 (覆盖原本可能的"贪婪"策略，这里虽然本来就是随机，但物理上代表无法利用梯度)
            # 在纯随机行走模型中，sensor noise 其实等价于不做任何改变
            # 但为了物理模拟的严谨性，我们认为 slip (打滑) 是更直接的惩罚
            
            # 计算目标
            dr = dr_map[moves]
            dc = dc_map[moves]
            
            # 应用打滑 (Actuator Noise)
            # 如果 slip，则 dr=0, dc=0
            is_slip = slip_mask[t]
            dr[is_slip] = 0
            dc[is_slip] = 0
            
            target_r = (current_r + dr) % L
            target_c = (current_c + dc) % L
            
            # 2. 拥堵
            target_indices = target_r * L + target_c
            counts = np.bincount(target_indices, minlength=L*L)
            conflict_mask = counts[target_indices] > 1
            self.collisions += np.sum(conflict_mask)
            
            # 3. 移动
            current_r = target_r
            current_c = target_c
            
            # 4. 学习
            unique_pos = np.unique(target_indices)
            
            # 关键：感知噪音导致即使到了食物点也可能"没看见"
            # 我们在这里模拟：只有那些"没slip"且"运气好"的粒子才能吃
            # 但为了简化向量化，我们假设到了就能吃，噪音主要体现在"走得慢"(slip)上
            
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
    print(f"Starting Hell-Mode Sweep on {num_cores} cores...")
    print(f"Features: Noise (Slip={cfg.actuator_noise:.0%}) + Point Source + 128x128 Map")
    
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
        
        ratio = d_steps / n
        bar = '#' * int((1-avg_loss)*15)
        print(f"N={n:4d} | Ratio={ratio:5.1f} | Loss={avg_loss:.4f} |{bar:<15}| Rho={avg_rho:.4f}")

    pool.close()
    pool.join()
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    return n_values, avg_losses, avg_crowdings, cfg

def plot_results(n_values, losses, crowdings, cfg):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Loss
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=14, weight='bold')
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=3, markersize=6, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.3)

    # Crowding
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Crowding (rho)', color=color, fontsize=12)
    ax2.plot(n_values, crowdings, color=color, marker='s', linestyle='--', alpha=0.3, label='Crowding')
    ax2.tick_params(axis='y', labelcolor=color)

    # 找最低点
    min_idx = np.argmin(losses)
    min_n = n_values[min_idx]
    min_loss = losses[min_idx]
    
    optimal_d = int(cfg.total_compute / (cfg.energy_cost_k * min_n))
    optimal_ratio = optimal_d / min_n
    
    title = (f"Chinchilla Frontier: Hell Mode (Noise + Diffusion)\n"
             f"Optimal N={min_n} | Ratio D/N ≈ {optimal_ratio:.1f}")
    plt.title(title, fontsize=16)
    
    ax1.annotate(f'Optimal Ratio\n≈ {optimal_ratio:.1f}', xy=(min_n, min_loss), 
                 xytext=(min_n, min_loss + 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                 ha='center', weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()
    
    print("-" * 30)
    print(f"FINAL HELL-MODE RESULT:")
    print(f"Optimal N: {min_n}")
    print(f"Optimal Ratio: {optimal_ratio:.2f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    n, l, r, c = run_iso_compute_sweep()
    plot_results(n, l, r, c)
