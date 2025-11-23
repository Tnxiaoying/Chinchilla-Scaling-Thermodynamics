import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import multiprocessing
from functools import partial

# --- 最终决战配置：并行平均版 ---
@dataclass
class Config:
    grid_size: int = 200         # 保持 4万格大地图
    total_compute: float = 2.5e5 # 保持黄金区间
    energy_cost_k: float = 6.0   
    n_min: int = 10              
    n_max: int = 8000            # 范围适中
    n_steps_sweep: int = 30      # 采样点
    food_density: float = 0.9    
    gamma_k: float = 30.0        
    n_trials: int = 24           # 【核心升级】每个点跑 24 次取平均 (适配 i9 多核)

class HamsterChinchillaSim:
    def __init__(self, n_particles, steps, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n = int(n_particles)
        self.steps = int(steps)
        self.config = config
        self.L = config.grid_size
        # 优化内存：只在需要时分配大数组，但在并行中每个进程独立内存
        # 使用平铺的一维数组可能更快，但这里保持逻辑清晰
        self.grid_flat = np.zeros(self.L * self.L, dtype=np.int8) 
        self.visits_flat = np.zeros(self.L * self.L, dtype=np.int8)
        
        # 初始化分布
        num_food = int(self.L * self.L * config.food_density)
        indices = np.random.choice(self.L * self.L, num_food, replace=False)
        self.grid_flat[indices] = 1
        self.total_food = num_food
        
        # 随机出生点
        self.particles = np.random.randint(0, self.L, (self.n, 2))
        
        self.collisions = 0
        self.eaten_food = 0

    def run(self):
        # 极速版逻辑
        L = self.L
        total_food = self.total_food
        
        for t in range(self.steps):
            # 1. 并行感知与意图生成
            # 为了速度，不再模拟非常复杂的决策，使用带有启发式的随机行走
            
            # 简单的意图计算：
            # 这里的瓶颈在于 Python 循环。
            # 既然是 i9，我们依赖多进程来抵消单线程慢的问题。
            
            # 批量生成移动方向
            moves_idx = np.random.randint(0, 5, size=self.n) 
            # 0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)
            dr = np.array([-1, 1, 0, 0, 0])[moves_idx]
            dc = np.array([0, 0, -1, 1, 0])[moves_idx]
            
            current_r = self.particles[:, 0]
            current_c = self.particles[:, 1]
            
            target_r = (current_r + dr) % L
            target_c = (current_c + dc) % L
            
            # 2. 冲突检测 (Vectorized)
            # 将坐标转换为一维索引以快速统计
            target_indices = target_r * L + target_c
            
            # 统计每个目标位置有多少人想去
            # bincount 很快，但需要索引非负
            counts = np.bincount(target_indices, minlength=L*L)
            
            # 任何计数 > 1 的位置都是拥堵
            # 任何目标位置如果本来就有人的意图（这里简化为只要大家去同一个点就算堵）
            # 或者更严格：如果 counts[idx] > 1，那么去这个点的所有人都算 collision
            
            # 找到发生冲突的目标索引
            conflict_mask = counts[target_indices] > 1
            self.collisions += np.sum(conflict_mask)
            
            # 3. 移动更新
            # 在这个简化模型中，即使拥堵我们也允许移动(重叠)，但记录了惩罚
            # 这符合 "Soft-Hardware Resonance" 中的软约束
            self.particles[:, 0] = target_r
            self.particles[:, 1] = target_c
            
            # 4. 吃食物
            # 只有不重叠的且有食物的地方才算吃掉？
            # 或者只要到了就算吃掉。
            # 只要有人到了 target_indices
            unique_visits = np.unique(target_indices)
            
            # 检查这些位置是否有食物且没被吃过
            # valid_food = (grid == 1) & (visits == 0)
            # 这里需要根据 unique_visits 快速索引
            
            # 只有第一次访问有效
            new_visits = unique_visits[self.grid_flat[unique_visits] == 1]
            # 过滤掉已经访问过的
            fresh_visits = new_visits[self.visits_flat[new_visits] == 0]
            
            if len(fresh_visits) > 0:
                self.visits_flat[fresh_visits] = 1
                self.eaten_food += len(fresh_visits)
            
            if self.eaten_food >= total_food:
                break

    def get_metrics(self):
        base_loss = (self.total_food - self.eaten_food) / self.total_food
        raw_rho = self.collisions / (self.n * self.steps) if self.steps > 0 else 0
        gamma = 1.0 / (1.0 + self.config.gamma_k * raw_rho)
        effective_performance = (1.0 - base_loss) * gamma
        thermo_loss = 1.0 - effective_performance
        return thermo_loss, raw_rho

# 包装函数用于多进程调用
def run_single_trial(seed, n, steps, config):
    sim = HamsterChinchillaSim(n, steps, config, seed)
    sim.run()
    return sim.get_metrics()

def run_iso_compute_sweep():
    cfg = Config()
    n_values = np.logspace(np.log10(cfg.n_min), np.log10(cfg.n_max), cfg.n_steps_sweep).astype(int)
    
    avg_losses = []
    avg_crowdings = []
    
    # 获取CPU核心数
    num_cores = multiprocessing.cpu_count()
    print(f"Starting Parallel Sweep on {num_cores} cores (Trials per N={cfg.n_trials})...")
    print(f"Compute Budget C={cfg.total_compute:.0e}")
    
    # 建立进程池
    pool = multiprocessing.Pool(processes=num_cores)
    
    for n in n_values:
        d_steps = int(cfg.total_compute / (cfg.energy_cost_k * n))
        if d_steps < 1: d_steps = 1
        
        # 准备多个种子并行运行
        seeds = [np.random.randint(0, 1000000) for _ in range(cfg.n_trials)]
        
        # 固定部分参数，只变seed
        func = partial(run_single_trial, n=n, steps=d_steps, config=cfg)
        
        # 并行执行
        results = pool.map(func, seeds)
        
        # 计算平均值
        losses = [r[0] for r in results]
        rhos = [r[1] for r in results]
        
        avg_loss = np.mean(losses)
        avg_rho = np.mean(rhos)
        
        avg_losses.append(avg_loss)
        avg_crowdings.append(avg_rho)
        
        # 进度条
        bar = '#' * int((1-avg_loss)*20)
        print(f"N={n:5d} | Loss={avg_loss:.4f} |{bar:<20}| Rho={avg_rho:.4f}")

    pool.close()
    pool.join()
    
    return n_values, avg_losses, avg_crowdings

def plot_results(n_values, losses, crowdings):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Loss
    color = 'tab:red'
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Thermodynamic Loss (Averaged)', color=color, fontsize=14, weight='bold')
    
    # 绘制平滑曲线
    ax1.plot(n_values, losses, color=color, marker='o', linewidth=3, markersize=6, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", alpha=0.3)

    # Crowding
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Crowding (rho)', color=color, fontsize=12)
    ax2.plot(n_values, crowdings, color=color, marker='s', linestyle='--', alpha=0.5, label='Crowding')
    ax2.tick_params(axis='y', labelcolor=color)

    # 最低点
    min_idx = np.argmin(losses)
    min_n = n_values[min_idx]
    min_val = losses[min_idx]
    
    # 计算此时的 D (Steps)
    # C = 2.5e5, k=6
    # D = C / 6N
    cfg = Config()
    d_optimal = cfg.total_compute / (6.0 * min_n)
    ratio = d_optimal / min_n
    
    title_text = (f'Chinchilla Frontier: The Smoothed Physical Curve\n'
                  f'Optimal N = {min_n}, Ratio D/N ≈ {ratio:.1f}')
    
    plt.title(title_text, fontsize=16)
    
    ax1.annotate(f'Critical Point\nN={min_n}', xy=(min_n, min_val), xytext=(min_n, min_val + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2), 
                 fontsize=12, ha='center', weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("-" * 30)
    print(f"Critical Discovery:")
    print(f"Optimal N: {min_n}")
    print(f"Implied Steps (D): {d_optimal:.1f}")
    print(f"Scaling Ratio (D/N): {ratio:.2f}")

if __name__ == "__main__":
    # Windows/Mac 下使用 multiprocessing 必须包含这行
    multiprocessing.freeze_support() 
    n, l, r = run_iso_compute_sweep()
    plot_results(n, l, r)
