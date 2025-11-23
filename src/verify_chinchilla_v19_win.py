import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# --- v19: Robustness Check (稳健性检验) ---
# 目标：验证 "内部最优点 N*" 是否随 "单粒子容量 (Capacity)" 发生符合物理直觉的移动。
# 预测：Capacity 越小 -> 需要的 N 越多 -> N* 右移。
#       Capacity 越大 -> 单个粒子越强 -> N* 左移。

@dataclass
class Config:
    grid_size: int = 80
    total_compute: float = 4e4       # 保持 v18 的参数
    energy_cost_k: float = 6.0
    
    n_min: int = 2
    n_max: int = 150
    n_steps_sweep: int = 30
    
    food_density: float = 0.9
    gamma_k: float = 10.0            # v18 的参数
    rho_threshold: float = 0.02
    
    n_trials: int = 20               # 稍微降低次数以节省总时间，足够看趋势
    
    # 扫描变量：这里设为列表，用于对比不同容量下的曲线
    capacities_to_scan: tuple = (40, 80, 160) 

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config, capacity):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.capacity = capacity     # 当前实验的容量限制
        self.L = config.grid_size
        
        # 1. 环境初始化
        self.total_slots = self.L * self.L
        # 使用一维数组优化性能
        self.grid = np.zeros(self.total_slots, dtype=np.int8)
        
        # 撒食物
        num_food = int(self.total_slots * config.food_density)
        indices = np.random.choice(self.total_slots, num_food, replace=False)
        self.grid[indices] = 1
        self.total_food = num_food
        
        # 2. 粒子状态
        # 记录每个粒子当前吃饱了没 (eaten_count)
        self.particle_eaten_counts = np.zeros(self.n, dtype=int)
        # 记录全局哪些位置被吃过了 (visited_map)
        self.visited_map = np.zeros(self.total_slots, dtype=np.int8)
        
        # 3. 位置初始化 (点源/中心)
        center = self.L // 2
        self.particles = np.zeros((self.n, 2), dtype=int) + center
        
        self.collisions = 0
        
    def run(self):
        L = self.L
        
        # 预生成随机移动 (Up, Down, Left, Right, Stay)
        # 0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)
        dr_map = np.array([-1, 1, 0, 0, 0], dtype=np.int8)
        dc_map = np.array([0, 0, -1, 1, 0], dtype=np.int8)
        all_moves = np.random.randint(0, 5, size=(self.steps, self.n), dtype=np.int8)
        
        # 10% Warmup
        warmup_steps = int(self.steps * 0.1)
        
        curr_r = self.particles[:, 0]
        curr_c = self.particles[:, 1]
        
        for t in range(self.steps):
            # --- A. 移动与拥堵 ---
            moves = all_moves[t]
            dr = dr_map[moves]
            dc = dc_map[moves]
            
            target_r = (curr_r + dr) % L
            target_c = (curr_c + dc) % L
            target_indices = target_r * L + target_c
            
            # 统计拥堵
            # 只有过了 warmup 才计入拥堵 (v18 逻辑)
            if t >= warmup_steps:
                counts = np.bincount(target_indices, minlength=self.total_slots)
                # 任何 > 1 的格子都算一次或多次冲突
                conflict_mask = counts[target_indices] > 1
                self.collisions += np.sum(conflict_mask)
            
            # 更新位置 (允许重叠，因为惩罚在 gamma 里)
            curr_r = target_r
            curr_c = target_c
            
            # --- B. 学习 (带容量限制) ---
            # 找到唯一的 (粒子ID, 位置ID) 对不太容易向量化，
            # 这里为了逻辑准确，采用"谁到了谁吃"的逻辑
            
            # 1. 找出当前步所有 有食物(1) 且 全局未访问(0) 的位置
            unique_pos_indices = np.unique(target_indices)
            
            # 筛选出潜在可吃的位置：Grid有食物 & 全局没被吃
            potential_mask = (self.grid[unique_pos_indices] == 1) & (self.visited_map[unique_pos_indices] == 0)
            eatable_positions = unique_pos_indices[potential_mask]
            
            if len(eatable_positions) > 0:
                # 哪些粒子还有肚子？
                hungry_mask = self.particle_eaten_counts < self.capacity
                hungry_indices = np.where(hungry_mask)[0]
                
                if len(hungry_indices) > 0:
                    hungry_targets = target_indices[hungry_indices]
                    
                    # 检查这些 hungry particles 是否在 eatable_positions 里
                    mask_in_eatable = np.isin(hungry_targets, eatable_positions)
                    successful_particles = hungry_indices[mask_in_eatable]
                    successful_targets = hungry_targets[mask_in_eatable]
                    
                    if len(successful_particles) > 0:
                        # 更新粒子肚子
                        np.add.at(self.particle_eaten_counts, successful_particles, 1)
                        
                        # 更新全局地图
                        self.visited_map[successful_targets] = 1

    def get_metrics(self):
        eaten_total = np.sum(self.visited_map)
        base_loss = 1.0 - (eaten_total / self.total_food)
        
        # 拥堵率 (跳过 warmup)
        valid_steps = max(1, int(self.steps * 0.9))
        avg_rho = self.collisions / (self.n * valid_steps)
        
        # 效率因子 Gamma
        penalty = max(0, avg_rho - self.config.rho_threshold)
        gamma = 1.0 / (1.0 + self.config.gamma_k * penalty)
        
        thermo_loss = 1.0 - (1.0 - base_loss) * gamma
        
        return thermo_loss, avg_rho

def run_sweep():
    cfg = Config()
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    n_values = np.unique(n_values)
    
    results = {} # key: capacity, val: (n_values, losses)
    
    print(f"Starting Robustness Check (v19)...")
    print(f"Scanning Capacities: {cfg.capacities_to_scan}")
    
    for cap in cfg.capacities_to_scan:
        print(f"\n--- Testing Capacity = {cap} ---")
        cap_losses = []
        
        for n in n_values:
            d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
            if d_steps < 1: d_steps = 1
            
            trials_loss = []
            for _ in range(cfg.n_trials):
                sim = HamsterChinchillaSim(n, d_steps, cfg, cap)
                sim.run()
                l, _ = sim.get_metrics()
                trials_loss.append(l)
            
            avg_loss = np.mean(trials_loss)
            cap_losses.append(avg_loss)
            
            # 简略进度条
            if n % 20 <= 2:
                print(f"N={n} Loss={avg_loss:.3f}", end=" | ")
                
        results[cap] = (n_values, cap_losses)
        
    return results

def plot_robustness(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['tab:blue', 'tab:red', 'tab:green']
    markers = ['^', 'o', 's']
    
    best_points = []
    
    for i, (cap, (n_vals, losses)) in enumerate(results.items()):
        # 找最低点
        min_idx = np.argmin(losses)
        min_n = n_vals[min_idx]
        min_loss = losses[min_idx]
        best_points.append((cap, min_n))
        
        label = f'Capacity={cap} (Opt N={min_n})'
        ax.plot(n_vals, losses, color=colors[i], marker=markers[i], 
                linewidth=2, markersize=6, label=label, alpha=0.8)
        
        # 标注最低点
        ax.plot(min_n, min_loss, marker='*', color='black', markersize=15, zorder=10)
        ax.vlines(min_n, 0, min_loss, color=colors[i], linestyle='--', alpha=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Particles (N)', fontsize=12)
    ax.set_ylabel('Thermodynamic Loss', fontsize=12)
    ax.set_title('Robustness Check: Optimal N shifts with Capacity', fontsize=14)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('robustness_check_v19.png')
    plt.show()
    
    print("\nSummary of Optima Shifts:")
    for cap, opt_n in best_points:
        print(f"Capacity {cap:3d} -> Optimal N ~ {opt_n}")

if __name__ == "__main__":
    data = run_sweep()
    plot_robustness(data)
