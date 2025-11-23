import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- 配置参数：世界扩容版 ---
@dataclass
class Config:
    grid_size: int = 200         # 保持宏大世界 (40,000格)
    total_compute: float = 8e4   # 【大砍刀】经费削减 80% (原 4e5 -> 8e4)
    energy_cost_k: float = 6.0   # Chinchilla Constant
    n_min: int = 10              # 
    n_max: int = 10000           # 
    n_steps_sweep: int = 35      # 稍微增加采样点让曲线更平滑
    food_density: float = 0.9    # 
    gamma_k: float = 30.0        # 【严刑峻法】进一步加强拥堵惩罚

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
        # 使用 Set 记录占据位置，加速大地图查找
        # 在大地图下，为了性能，依然使用 simplified occupied map
        
        for t in range(self.steps):
            # 当前帧的占据情况
            occupied_set = set((p[0], p[1]) for p in self.particles)
            
            new_positions = []
            current_collisions = 0
            
            # 简单的碰撞预测
            next_occupied_counts = {} 
            
            # 预计算所有人的意图
            intentions = []
            for i in range(self.n):
                r, c = self.particles[i]
                
                # 感知
                moves = [(-1,0), (1,0), (0,-1), (0,1), (0,0)]
                best_move = (0,0)
                found_food = False
                np.random.shuffle(moves)
                
                for dr, dc in moves:
                    nr, nc = (r + dr) % self.L, (c + dc) % self.L
                    # 只去没去过且有食物的地方
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

            # 执行移动
            for i in range(self.n):
                r, c = self.particles[i]
                target_r, target_c = intentions[i]
                
                # 判定逻辑：
                # 1. 如果目标点有多人抢，视为拥堵
                # 2. 如果目标点本来就有人(且不是自己)，视为拥堵
                # 3. 简化：只要有人(包括自己移动后的重叠)，就算 collision
                
                collision = False
                # 无论目标点是否被抢，只要有人在就是拥挤的体现
                if next_occupied_counts[(target_r, target_c)] > 1:
                    current_collisions += 1
                    collision = True
                
                # 如果发生严重冲突，留在原地 (模拟阻塞)
                # 这里为了模拟更流畅，允许移动但记录 collision
                # 或者严格模式：
                # if collision: target_r, target_c = r, c 
                
                # 更新
                new_positions.append([target_r, target_c])
                
                # 吃食物
                if self.grid[target_r, target_c] == 1 and self.visits[target_r, target_c] == 0:
                    self.visits[target_r, target_c] = 1
                    self.eaten_food += 1

            self.particles = np.array(new_positions)
            self.collisions += current_collisions
            
            # 提前终止检查
            if self.eaten_food >= self.total_food:
                break

    def get_metrics(self):
        base_loss = (self.total_food - self.eaten_food) / self.total_food
        
        # 计算拥堵度 rho
        raw_rho = self.collisions / (self.n * self.steps) if self.steps > 0 else 0
        
        # Gamma 效率修正
        gamma = 1.0 / (1.0 + self.config.gamma_k * raw_rho)
        
        # 热力学 Loss
        effective_performance = (1.0 - base_loss) * gamma
        thermo_loss = 1.0 - effective_performance
        
        return thermo_loss, raw_rho

def run_iso_compute_sweep():
    cfg = Config()
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    losses = []
    crowdings = []
    
    print(f"Starting Large-Scale Iso-Compute Sweep (C={cfg.total_compute:.0e}, Grid={cfg.grid_size}x{cfg.grid_size})...")
    
    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1: d_steps = 1
        
        sim = HamsterChinchillaSim(n, d_steps, cfg)
        sim.run()
        loss, rho = sim.get_metrics()
        
        losses.append(loss)
        crowdings.append(rho)
        print(f"N={n:5d}, Steps={d_steps:5d} | Loss={loss:.4f}, Rho={rho:.4f}")

    return n_values, losses, crowdings

def plot_results(n_values, losses, crowdings):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Loss
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)')
    ax1.set_ylabel('Thermodynamic Loss', color=color, fontsize=12, weight='bold')
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.2)

    # Crowding
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Crowding (rho)', color=color)
    ax2.plot(n_values, crowdings, color=color, marker='s', linestyle='--', alpha=0.5, label='Crowding')
    ax2.tick_params(axis='y', labelcolor=color)

    # 最低点
    min_idx = np.argmin(losses)
    min_n = n_values[min_idx]
    min_val = losses[min_idx]
    
    plt.title('Chinchilla Frontier: The U-Shape Discovery\n(Scale-up World Simulation)', fontsize=14)
    ax1.annotate(f'Optimal N: {min_n}', xy=(min_n, min_val), xytext=(min_n, min_val + 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.show()
    return min_n

if __name__ == "__main__":
    n, l, r = run_iso_compute_sweep()
    plot_results(n, l, r)
