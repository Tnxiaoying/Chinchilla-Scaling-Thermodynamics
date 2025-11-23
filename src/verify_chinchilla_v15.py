import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# --- v15: 生态平衡版 (Survivable Zone) ---
# 目标：让最优解由于"恰好够用"而脱颖而出。
# 之前的问题：难度太大，所有 N 都失败了 (Loss > 0.8)。
# 修正：降低地图尺寸，让中间 N 的 Loss 能降到 0.4 左右，从而显现 U 型。

@dataclass
class Config:
    grid_size: int = 64          # 【回调】回到 64x64，让任务变得"可完成"
    total_compute: float = 1.2e5 # 预算适配小地图
    energy_cost_k: float = 6.0   
    n_min: int = 2               
    n_max: int = 200             # 扫描区间
    n_steps_sweep: int = 35      
    food_density: float = 0.9    
    gamma_k: float = 40.0        # 拥堵惩罚保持高压
    n_trials: int = 50           # 保持饱和采样
    init_radius: int = 0         # 点源
    sensor_noise: float = 0.15   # 稍微降低噪音，让大家有机会吃饭
    actuator_noise: float = 0.05 

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.L = config.grid_size
        
        self.grid_flat = np.zeros(self.L * self.L, dtype=np.uint8)
        self.visits_flat = np.zeros(self.L * self.L, dtype=np.uint8)
        
        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid_flat[indices] = 1
        self.total_food = num_food
        
        center = self.L // 2
        self.particles = np.zeros((self.n, 2), dtype=int) + center
        
        self.collisions = 0
        self.eaten_food = 0

    def run(self):
        L = self.L
        total_food = self.total_food
        
        all_moves_idx = np.random.randint(0, 5, size=(self.steps, self.n), dtype=np.int8)
        dr_map = np.array([-1, 1, 0, 0, 0], dtype=np.int8)
        dc_map = np.array([0, 0, -1, 1, 0], dtype=np.int8)
        
        sensor_noise_p = self.config.sensor_noise
        slip_p = self.config.actuator_noise
        rand_floats = np.random.rand(self.steps, self.n * 2).astype(np.float32)
        
        current_r = self.particles[:, 0]
        current_c = self.particles[:, 1]
        
        for t in range(self.steps):
            moves = all_moves_idx[t]
            dr = dr_map[moves]
            dc = dc_map[moves]
            
            is_slip = rand_floats[t, :self.n] < slip_p
            dr[is_slip] = 0
            dc[is_slip] = 0
            
            target_r = (current_r + dr) % L
            target_c = (current_c + dc) % L
            
            target_indices = target_r * L + target_c
            counts = np.bincount(target_indices, minlength=L*L)
            conflict_mask = counts[target_indices] > 1
            self.collisions += np.sum(conflict_mask)
            
            current_r = target_r
            current_c = target_c
            
            unique_pos = np.unique(target_indices)
            potential_new = unique_pos[(self.grid_flat[unique_pos] == 1) & (self.visits_flat[unique_pos] == 0)]
            
            if len(potential_new) > 0:
                # 简单的感知噪音
                if sensor_noise_p > 0:
                    success_mask = np.random.rand(len(potential_new)) > sensor_noise_p
                    real_new = potential_new[success_mask]
                else:
                    real_new = potential_new
                
                if len(real_new) > 0:
                    self.visits_flat[real_new] = 1
                    self.eaten_food += len(real_new)
            
            if self.eaten_food >= total_food:
                break

    def get_metrics(self):
        base_loss = (self.total_food - self.eaten_food) / self.total_food
        raw_rho = self.collisions / (self.n * self.steps) if self.steps > 0 else 0
        gamma = 1.0 / (1.0 + self.config.gamma_k * raw_rho)
        effective_performance = (1.0 - base_loss) * gamma
        thermo_loss = 1.0 - effective_performance
        return thermo_loss, raw_rho

def run_iso_compute_sweep():
    cfg = Config()
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    n_values = np.unique(n_values)
    
    avg_losses = []
    avg_crowdings = []
    
    print(f"Starting Survivable Zone Sweep (N=2..200)...")
    start_time = time.time()
    
    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1: d_steps = 1
        
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
        bar = '#' * int((1-avg_loss)*20)
        print(f"N={n:3d} | Ratio={ratio:5.1f} | Loss={avg_loss:.4f} |{bar:<20}|")

    print(f"Total time: {time.time() - start_time:.2f}s")
    return n_values, avg_losses, avg_crowdings, cfg

def plot_results(n_values, losses, crowdings, cfg):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=14, weight='bold')
    
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=3, markersize=8, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.4)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Crowding (rho)', color=color, fontsize=12)
    ax2.plot(n_values, crowdings, color=color, marker='s', linestyle='--', alpha=0.3, label='Crowding')
    ax2.tick_params(axis='y', labelcolor=color)

    min_idx = np.argmin(losses)
    min_n = n_values[min_idx]
    min_loss = losses[min_idx]
    
    optimal_d = int(cfg.total_compute / (cfg.energy_cost_k * min_n))
    optimal_ratio = optimal_d / min_n
    
    title = (f"Chinchilla Frontier: The Survivable Valley\n"
             f"Optimal N={min_n} | Ratio D/N ≈ {optimal_ratio:.1f}")
    plt.title(title, fontsize=16)
    
    ax1.annotate(f'THE VALLEY\nRatio ≈ {optimal_ratio:.1f}', xy=(min_n, min_loss), 
                 xytext=(min_n, min_loss + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=3),
                 ha='center', weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n, l, r, c = run_iso_compute_sweep()
    plot_results(n, l, r, c)
