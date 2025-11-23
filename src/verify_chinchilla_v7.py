import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- 针对 i9 CPU 优化的配置 ---
@dataclass
class Config:
    grid_size: int = 200         # 保持 40,000 格的大地图
    total_compute: float = 2.5e5 # 【关键调整】黄金区间：25万算力
    energy_cost_k: float = 6.0   
    n_min: int = 10              
    n_max: int = 10000           
    n_steps_sweep: int = 40      # 【i9 专属】提高采样密度，画出丝滑曲线
    food_density: float = 0.9    
    gamma_k: float = 30.0        # 保持严厉的拥堵惩罚

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.L = config.grid_size
        self.grid = np.zeros((self.L, self.L)) 
        self.visits = np.zeros((self.L, self.L))
        
        # 初始化分布
        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid.flat[indices] = 1
        self.total_food = num_food
        
        # 随机出生点
        self.particles = np.random.randint(0, self.L, (self.n, 2))
        
        self.collisions = 0
        self.eaten_food = 0

    def run(self):
        # 针对 i9 优化：无需过度简化的逻辑，直接跑完整模拟
        
        for t in range(self.steps):
            # 意图计算
            intentions = []
            next_occupied_counts = {} 
            
            # 1. 并行感知 (模拟)
            for i in range(self.n):
                r, c = self.particles[i]
                
                moves = [(-1,0), (1,0), (0,-1), (0,1), (0,0)]
                np.random.shuffle(moves)
                best_move = (0,0)
                found_food = False
                
                for dr, dc in moves:
                    nr, nc = (r + dr) % self.L, (c + dc) % self.L
                    if self.grid[nr, nc] == 1 and self.visits[nr, nc] == 0:
                        best_move = (dr, dc)
                        found_food = True
                        break
                
                if not found_food:
                    idx = np.random.randint(0, len(moves))
                    best_move = moves[idx]
                
                nr, nc = (r + best_move[0]) % self.L, (c + best_move[1]) % self.L
                intentions.append((nr, nc))
                next_occupied_counts[(nr, nc)] = next_occupied_counts.get((nr, nc), 0) + 1

            # 2. 移动执行与冲突检测
            new_positions = []
            current_collisions = 0
            
            for i in range(self.n):
                target_r, target_c = intentions[i]
                
                # 只要目标点有多人想去，或者已经有人，都算拥堵贡献
                if next_occupied_counts[(target_r, target_c)] > 1:
                    current_collisions += 1
                
                # 移动
                new_positions.append([target_r, target_c])
                
                # 吃食物
                if self.grid[target_r, target_c] == 1 and self.visits[target_r, target_c] == 0:
                    self.visits[target_r, target_c] = 1
                    self.eaten_food += 1

            self.particles = np.array(new_positions)
            self.collisions += current_collisions
            
            if self.eaten_food >= self.total_food:
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
    losses = []
    crowdings = []
    
    print(f"Starting Goldilocks Sweep on i9 (C={cfg.total_compute:.0e})...")
    
    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1: d_steps = 1
        
        sim = HamsterChinchillaSim(n, d_steps, cfg)
        sim.run()
        loss, rho = sim.get_metrics()
        
        losses.append(loss)
        crowdings.append(rho)
        # 打印进度条，让 i9 跑得更有感觉
        bar = '#' * int((1-loss)*20)
        print(f"N={n:5d}, Steps={d_steps:5d} | Loss={loss:.4f} |{bar:<20}| Rho={rho:.4f}")

    return n_values, losses, crowdings

def plot_results(n_values, losses, crowdings):
    fig, ax1 = plt.subplots(figsize=(12, 7)) # 更大的图
    
    # Loss
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=14, weight='bold')
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=3, markersize=8, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.3)

    # Crowding
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Crowding (rho)', color=color, fontsize=12)
    ax2.plot(n_values, crowdings, color=color, marker='s', linestyle='--', alpha=0.6, label='Crowding')
    ax2.tick_params(axis='y', labelcolor=color)

    # 最低点
    min_idx = np.argmin(losses)
    min_n = n_values[min_idx]
    min_val = losses[min_idx]
    
    plt.title(f'Chinchilla Frontier: The "Goldilocks" Zone\n(Optimal N = {min_n})', fontsize=16)
    
    # 高亮最低点
    ax1.annotate(f'Optimal\nN={min_n}', xy=(min_n, min_val), xytext=(min_n, min_val + 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2), 
                 fontsize=12, ha='center', weight='bold')
    
    plt.tight_layout()
    plt.show()
    return min_n

if __name__ == "__main__":
    n, l, r = run_iso_compute_sweep()
    plot_results(n, l, r)
