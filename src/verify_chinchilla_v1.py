import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

# --- 配置参数 ---
@dataclass
class Config:
    grid_size: int = 64          # L x L 网格 (物理空间限制)
    total_compute: float = 1e7   # C: 总热力学预算
    energy_cost_k: float = 6.0   # k: 单粒子单步能耗 (Forward/Backward)
    n_min: int = 10              # 最小粒子数 (Under-parameterized)
    n_max: int = 1000            # 最大粒子数 (Over-parameterized)
    n_steps_sweep: int = 20      # 扫描采样点数
    food_density: float = 0.8    # 初始信息/食物覆盖率

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config):
        self.n = int(n_particles)
        self.steps = int(steps)
        self.L = config.grid_size
        self.grid = np.zeros((self.L, self.L)) # 0: Empty, 1: Food
        self.visits = np.zeros((self.L, self.L)) # 记录访问情况
        
        # 初始化信息(食物)分布
        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid.flat[indices] = 1
        self.total_food = num_food
        
        # 初始化粒子位置
        self.particles = np.random.randint(0, self.L, (self.n, 2))
        
        # 统计量
        self.collisions = 0
        self.eaten_food = 0

    def run(self):
        # 简单的贪婪+随机行走策略
        # 如果周围有食物，趋向食物；否则随机行走
        # 拥堵机制：目标格子若被占据，则不动并记录碰撞
        
        occupied_map = np.zeros((self.L, self.L))
        for p in self.particles:
            occupied_map[p[0], p[1]] += 1

        for t in range(self.steps):
            new_positions = []
            current_collisions = 0
            
            # 重新计算占据图 (简单的同步更新近似)
            next_occupied_map = np.zeros_like(occupied_map)
            
            for i in range(self.n):
                r, c = self.particles[i]
                
                # --- 1. 感知与决策 ---
                # 寻找邻域内的最佳移动 (贪婪策略模拟梯度下降)
                moves = [(-1,0), (1,0), (0,-1), (0,1), (0,0)]
                best_move = (0,0)
                found_food = False
                
                # 简单乱序以避免方向偏差
                np.random.shuffle(moves)
                
                for dr, dc in moves:
                    nr, nc = (r + dr) % self.L, (c + dc) % self.L
                    if self.grid[nr, nc] == 1 and self.visits[nr, nc] == 0:
                        best_move = (dr, dc)
                        found_food = True
                        break
                
                if not found_food:
                    # 没有直接食物，随机漫步 (Exploration)
                    idx = np.random.randint(0, len(moves))
                    best_move = moves[idx]

                # --- 2. 移动与拥堵判定 ---
                dr, dc = best_move
                nr, nc = (r + dr) % self.L, (c + dc) % self.L
                
                # 检查目标位置在上一帧是否被占据 (简化版拥堵物理)
                # 如果目标位置有多人竞争或已被占据，视为“软拥堵”，增加摩擦
                # 这里简化为：如果目标位有人，则有概率移动失败
                
                if occupied_map[nr, nc] > 0 and (nr != r or nc != c):
                    # 发生冲突/拥堵
                    current_collisions += 1
                    # 拥堵导致停滞 (Reject move)
                    nr, nc = r, c
                
                new_positions.append([nr, nc])
                next_occupied_map[nr, nc] += 1
                
                # --- 3. 吃食物 (Learning) ---
                if self.grid[nr, nc] == 1 and self.visits[nr, nc] == 0:
                    self.visits[nr, nc] = 1
                    self.eaten_food += 1

            self.particles = np.array(new_positions)
            occupied_map = next_occupied_map
            self.collisions += current_collisions
            
            # 如果所有食物都被吃光，提前终止 (可选，但为了严格对比保持steps运行)
            if self.eaten_food >= self.total_food:
                break

    def get_metrics(self):
        loss = (self.total_food - self.eaten_food) / self.total_food
        # 归一化拥堵率：平均每个粒子每步遇到的冲突数
        crowding_rho = self.collisions / (self.n * self.steps) if self.steps > 0 else 0
        return loss, crowding_rho

def run_iso_compute_sweep():
    cfg = Config()
    
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    losses = []
    crowdings = []
    steps_list = []
    
    print(f"Starting Iso-Compute Sweep (C={cfg.total_compute:.0e})...")
    
    for n in n_values:
        # 核心 Chinchilla 约束方程: D = C / (6N)
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        
        # 只有当步数足够且粒子数合理时才运行
        if d_steps < 1: 
            d_steps = 1
        
        sim = HamsterChinchillaSim(n, d_steps, cfg)
        sim.run()
        
        loss, rho = sim.get_metrics()
        
        losses.append(loss)
        crowdings.append(rho)
        steps_list.append(d_steps)
        
        print(f"N={n:4d}, Steps={d_steps:5d} | Loss={loss:.4f}, Rho={rho:.4f}")

    return n_values, losses, crowdings

def plot_results(n_values, losses, crowdings):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss 曲线 (左轴)
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)')
    ax1.set_ylabel('Loss (Unlearned Info)', color=color)
    ax1.plot(n_values, losses, color=color, marker='o', label='Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # 绘制 Crowding 曲线 (右轴)
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Crowding Factor (rho)', color=color)
    ax2.plot(n_values, crowdings, color=color, marker='s', linestyle='--', label='Crowding', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)

    # 寻找 Loss 最低点
    min_loss_idx = np.argmin(losses)
    min_loss_n = n_values[min_loss_idx]
    min_loss_val = losses[min_loss_idx]
    
    plt.title('Chinchilla Frontier: Thermodynamic Criticality\n(Iso-Compute Sweep)')
    
    # 标注最优 N
    ax1.annotate(f'Optimal N: {min_loss_n}', xy=(min_loss_n, min_loss_val), 
                 xytext=(min_loss_n, min_loss_val + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.show()
    
    return min_loss_n

if __name__ == "__main__":
    # 运行实验
    n_vals, loss_vals, rho_vals = run_iso_compute_sweep()
    
    # 绘图
    opt_n = plot_results(n_vals, loss_vals, rho_vals)
    
    print("-" * 30)
    print(f"Experiment Complete.")
    print(f"Optimal Particle Count (N): {opt_n}")
    # 这里我们可以手动检查对应的 Crowding 是否处于拐点
