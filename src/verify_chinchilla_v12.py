import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# --- v12: i9 单核极速版 (避开 Multiprocessing Pickling 问题) ---
# 保持所有地狱难度设定，但回归单线程以确保稳定性。

@dataclass
class Config:
    grid_size: int = 128         # 地狱级地图
    total_compute: float = 8e5   # 预算
    energy_cost_k: float = 6.0   
    n_min: int = 2               
    n_max: int = 5000            
    n_steps_sweep: int = 35      # 采样点适中
    food_density: float = 0.9    
    gamma_k: float = 30.0        
    n_trials: int = 8            # 单核跑8次平均足够平滑了 (依靠 i9 高主频)
    init_radius: int = 0         # 点源初始化
    sensor_noise: float = 0.25   # 感知噪音
    actuator_noise: float = 0.10 # 行动噪音

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config):
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
        
        center = self.L // 2
        self.particles = np.zeros((self.n, 2), dtype=int) + center
        
        self.collisions = 0
        self.eaten_food = 0

    def run(self):
        L = self.L
        total_food = self.total_food
        noise_p = self.config.sensor_noise
        slip_p = self.config.actuator_noise
        
        all_moves_idx = np.random.randint(0, 5, size=(self.steps, self.n))
        dr_map = np.array([-1, 1, 0, 0, 0])
        dc_map = np.array([0, 0, -1, 1, 0])
        
        sensor_fail_mask = np.random.rand(self.steps, self.n) < noise_p
        slip_mask = np.random.rand(self.steps, self.n) < slip_p
        
        current_r = self.particles[:, 0]
        current_c = self.particles[:, 1]
        
        for t in range(self.steps):
            moves = all_moves_idx[t]
            dr = dr_map[moves]
            dc = dc_map[moves]
            
            # Actuator Noise (打滑)
            is_slip = slip_mask[t]
            dr[is_slip] = 0
            dc[is_slip] = 0
            
            target_r = (current_r + dr) % L
            target_c = (current_c + dc) % L
            
            # Jamming
            target_indices = target_r * L + target_c
            counts = np.bincount(target_indices, minlength=L*L)
            # 简单的冲突判定：只要那个格子有>1人想去
            conflict_mask = counts[target_indices] > 1
            self.collisions += np.sum(conflict_mask)
            
            current_r = target_r
            current_c = target_c
            
            # Learning (需克服 Sensor Noise)
            # 这里简化：只要不slip且到了，就算可能吃到
            # Sensor noise 在这里隐式体现为：如果没有足够的步数去探索，你就吃不到
            
            unique_pos = np.unique(target_indices)
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

def run_iso_compute_sweep():
    cfg = Config()
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    
    avg_losses = []
    avg_crowdings = []
    
    print(f"Starting Single-Core Speed Run (i9 Mode)...")
    start_time = time.time()
    
    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1: d_steps = 1
        
        losses = []
        rhos = []
        
        # 串行跑 n_trials 次
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
        bar = '#' * int((1-avg_loss)*15)
        print(f"N={n:4d} | Ratio={ratio:5.1f} | Loss={avg_loss:.4f} |{bar:<15}|")

    print(f"Total time: {time.time() - start_time:.2f}s")
    return n_values, avg_losses, avg_crowdings, cfg

def plot_results(n_values, losses, crowdings, cfg):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=14, weight='bold')
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=3, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.3)

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
    
    title = (f"Chinchilla Frontier: Hell Mode Result\n"
             f"Optimal N={min_n} | Ratio D/N ≈ {optimal_ratio:.1f}")
    plt.title(title, fontsize=16)
    
    ax1.annotate(f'Optimal Ratio\n≈ {optimal_ratio:.1f}', xy=(min_n, min_loss), 
                 xytext=(min_n, min_loss + 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                 ha='center', weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n, l, r, c = run_iso_compute_sweep()
    plot_results(n, l, r, c)
