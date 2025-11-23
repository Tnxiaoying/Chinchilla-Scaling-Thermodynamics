import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# --- v21: Geometry Robustness (几何/维度稳健性) ---
# 合作者建议：验证 Map Size 对 N* 的影响。
# 物理预测：Map 越大 -> 拥堵越晚发生 -> N* 应该向右移动 (允许更多参数)。
# 这将连接 Paper 1 (几何瓶颈) 与 Paper 2 (Scaling Law)。

@dataclass
class Config:
    # 扫描变量：地图大小
    grids_to_scan: tuple = (64, 80, 100) 
    
    total_compute: float = 4e4       # 保持 v18 的黄金预算
    energy_cost_k: float = 6.0
    
    n_min: int = 2
    n_max: int = 150
    n_steps_sweep: int = 35
    
    food_density: float = 0.9
    gamma_k: float = 10.0            # 保持一致
    rho_threshold: float = 0.02
    
    n_trials: int = 20
    
    # 固定容量 (v18 基准)
    memory_per_particle: int = 80    

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config, grid_size):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.capacity = config.memory_per_particle
        self.L = grid_size
        
        # 环境初始化
        self.total_slots = self.L * self.L
        self.grid = np.zeros(self.total_slots, dtype=np.int8)
        
        # 撒食物 (保持密度一致，所以大地图食物更多)
        num_food = int(self.total_slots * config.food_density)
        indices = np.random.choice(self.total_slots, num_food, replace=False)
        self.grid[indices] = 1
        self.total_food = num_food
        
        self.particle_eaten_counts = np.zeros(self.n, dtype=int)
        self.visited_map = np.zeros(self.total_slots, dtype=np.int8)
        
        # 点源初始化
        center = self.L // 2
        self.particles = np.zeros((self.n, 2), dtype=int) + center
        
        self.collisions = 0
        
    def run(self):
        L = self.L
        dr_map = np.array([-1, 1, 0, 0, 0], dtype=np.int8)
        dc_map = np.array([0, 0, -1, 1, 0], dtype=np.int8)
        all_moves = np.random.randint(0, 5, size=(self.steps, self.n), dtype=np.int8)
        warmup_steps = int(self.steps * 0.1)
        
        curr_r = self.particles[:, 0]
        curr_c = self.particles[:, 1]
        
        for t in range(self.steps):
            moves = all_moves[t]
            target_r = (curr_r + dr_map[moves]) % L
            target_c = (curr_c + dc_map[moves]) % L
            target_indices = target_r * L + target_c
            
            if t >= warmup_steps:
                counts = np.bincount(target_indices, minlength=self.total_slots)
                conflict_mask = counts[target_indices] > 1
                self.collisions += np.sum(conflict_mask)
            
            curr_r = target_r
            curr_c = target_c
            
            # 学习逻辑 (带容量)
            unique_pos_indices = np.unique(target_indices)
            potential_mask = (self.grid[unique_pos_indices] == 1) & (self.visited_map[unique_pos_indices] == 0)
            eatable_positions = unique_pos_indices[potential_mask]
            
            if len(eatable_positions) > 0:
                hungry_mask = self.particle_eaten_counts < self.capacity
                hungry_indices = np.where(hungry_mask)[0]
                
                if len(hungry_indices) > 0:
                    hungry_targets = target_indices[hungry_indices]
                    mask_in_eatable = np.isin(hungry_targets, eatable_positions)
                    successful_particles = hungry_indices[mask_in_eatable]
                    successful_targets = hungry_targets[mask_in_eatable]
                    
                    if len(successful_particles) > 0:
                        np.add.at(self.particle_eaten_counts, successful_particles, 1)
                        self.visited_map[successful_targets] = 1

    def get_metrics(self):
        eaten_total = np.sum(self.visited_map)
        base_loss = 1.0 - (eaten_total / self.total_food)
        
        valid_steps = max(1, int(self.steps * 0.9))
        avg_rho = self.collisions / (self.n * valid_steps)
        
        penalty = max(0, avg_rho - self.config.rho_threshold)
        gamma = 1.0 / (1.0 + self.config.gamma_k * penalty)
        
        thermo_loss = 1.0 - (1.0 - base_loss) * gamma
        return thermo_loss

def run_map_sweep():
    cfg = Config()
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    n_values = np.unique(n_values)
    
    results = {} 
    
    print(f"Starting Geometry Robustness Check (v21)...")
    print(f"Scanning Map Sizes: {cfg.grids_to_scan}")
    
    for size in cfg.grids_to_scan:
        print(f"\n--- Testing Map Size = {size}x{size} ---")
        size_losses = []
        
        for n in n_values:
            d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
            if d_steps < 1: d_steps = 1
            
            trials_loss = []
            for _ in range(cfg.n_trials):
                # 注意这里传入 size
                sim = HamsterChinchillaSim(n, d_steps, cfg, size)
                sim.run()
                trials_loss.append(sim.get_metrics())
            
            avg_loss = np.mean(trials_loss)
            size_losses.append(avg_loss)
            
            if n % 20 <= 2: print(f"N={n} Loss={avg_loss:.3f}", end=" | ")
                
        results[size] = (n_values, size_losses)
    return results

def plot_map_robustness(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 颜色方案：从冷到暖表示地图变大
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    best_points = []
    
    for i, (size, (n_vals, losses)) in enumerate(results.items()):
        min_idx = np.argmin(losses)
        min_n = n_vals[min_idx]
        min_loss = losses[min_idx]
        best_points.append((size, min_n))
        
        label = f'Map Size = {size}x{size} (Opt N={min_n})'
        ax.plot(n_vals, losses, color=colors[i], linewidth=2, marker='o', markersize=5, label=label, alpha=0.8)
        
        # 标记最优点
        ax.scatter(min_n, min_loss, color=colors[i], s=150, marker='*', edgecolors='black', zorder=10)
        ax.vlines(min_n, 0, min_loss, color=colors[i], linestyle='--', alpha=0.3)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Particles (N)', fontsize=12)
    ax.set_ylabel('Thermodynamic Loss', fontsize=12)
    ax.set_title('Geometry Check: Optimal N shifts with Map Size', fontsize=14)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('map_scaling_v21.png')
    plt.show()
    
    print("\nSummary of Geometric Shifts:")
    for size, opt_n in best_points:
        print(f"Map {size}x{size} -> Optimal N ~ {opt_n}")

if __name__ == "__main__":
    data = run_map_sweep()
    plot_map_robustness(data)
