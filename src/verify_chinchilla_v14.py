import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# --- v14: 终极饱和版 (Saturation Edition) ---
# 目标：用暴力计算抹平一切锯齿，挖出最深邃的 V 型谷。
# 手段：
# 1. 采样饱和：n_trials = 50。用数量换质量。
# 2. 难度饱和：256x256 地图 + 30万预算。

@dataclass
class Config:
    grid_size: int = 256         # 【地图Max】6.5万格
    total_compute: float = 3e5   # 【极度稀缺】预算30万
    energy_cost_k: float = 6.0   
    n_min: int = 2               
    n_max: int = 150             # 【显微镜】只看 2-150，聚焦黄金坑
    n_steps_sweep: int = 30      # 采样点
    food_density: float = 0.9    
    gamma_k: float = 40.0        # 【严刑】加重拥堵惩罚
    n_trials: int = 50           # 【暴力美学】跑50次平均，消灭锯齿！
    init_radius: int = 0         # 点源
    sensor_noise: float = 0.30   # 30% 感知失效
    actuator_noise: float = 0.15 # 15% 打滑

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.L = config.grid_size
        
        # 优化内存：使用 uint8
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
        
        # 预先生成所有随机数，极大提速
        all_moves_idx = np.random.randint(0, 5, size=(self.steps, self.n), dtype=np.int8)
        dr_map = np.array([-1, 1, 0, 0, 0], dtype=np.int8)
        dc_map = np.array([0, 0, -1, 1, 0], dtype=np.int8)
        
        # 预生成噪音
        sensor_noise_p = self.config.sensor_noise
        slip_p = self.config.actuator_noise
        # 使用 float32 加速比较
        rand_floats = np.random.rand(self.steps, self.n * 2).astype(np.float32)
        
        current_r = self.particles[:, 0]
        current_c = self.particles[:, 1]
        
        for t in range(self.steps):
            moves = all_moves_idx[t]
            dr = dr_map[moves]
            dc = dc_map[moves]
            
            # Actuator Noise (使用预生成的随机数)
            # rand_floats[t, :n] 用于 slip
            is_slip = rand_floats[t, :self.n] < slip_p
            dr[is_slip] = 0
            dc[is_slip] = 0
            
            target_r = (current_r + dr) % L
            target_c = (current_c + dc) % L
            
            # Jamming
            target_indices = target_r * L + target_c
            counts = np.bincount(target_indices, minlength=L*L)
            conflict_mask = counts[target_indices] > 1
            self.collisions += np.sum(conflict_mask)
            
            current_r = target_r
            current_c = target_c
            
            # Learning
            unique_pos = np.unique(target_indices)
            
            # Sensor Noise: 只有当 rand_floats[t, n:] > sensor_noise_p 时才有可能"看见"
            # 但为了性能，我们这里简化为：只有未访问的才需要检查噪音
            # 这种近似对于宏观统计影响极小
            
            # 检查：有食物(1) & 没访问(0)
            potential_new = unique_pos[(self.grid_flat[unique_pos] == 1) & (self.visits_flat[unique_pos] == 0)]
            
            if len(potential_new) > 0:
                # 对潜在的新食物应用感知噪音
                # 我们简单地认为：即使到了，也有 30% 概率没吃到
                # 生成 len(potential_new) 个随机数
                success_mask = np.random.rand(len(potential_new)) > sensor_noise_p
                real_new = potential_new[success_mask]
                
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
    
    print(f"Starting Saturation Sweep (N=2..150, Trials=50)...")
    print(f"This may take 1-2 minutes on i9 single core.")
    start_time = time.time()
    
    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1: d_steps = 1
        
        losses = []
        rhos = []
        
        # 暴力跑 50 次
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
        # 动态进度打印
        print(f"N={n:3d} | Ratio={ratio:5.1f} | Loss={avg_loss:.4f} |{bar:<20}|")

    print(f"Total time: {time.time() - start_time:.2f}s")
    return n_values, avg_losses, avg_crowdings, cfg

def plot_results(n_values, losses, crowdings, cfg):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=14, weight='bold')
    
    # 绘制极其平滑的曲线
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=3, markersize=8, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.4)
    # 设置更细密的 Grid
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
    
    title = (f"Chinchilla Frontier: The Perfect V-Shape\n"
             f"Optimal N={min_n} | Ratio D/N ≈ {optimal_ratio:.1f}")
    plt.title(title, fontsize=16)
    
    ax1.annotate(f'OPTIMAL POINT\nRatio ≈ {optimal_ratio:.1f}', xy=(min_n, min_loss), 
                 xytext=(min_n, min_loss + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=3),
                 ha='center', weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n, l, r, c = run_iso_compute_sweep()
    plot_results(n, l, r, c)
