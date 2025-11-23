import numpy as np
import matplotlib.pyplot as plt

# --- v20 Final: Theoretical Abstraction (Publication Ready) ---
# Target: Generate a clean theoretical plot showing N* shifts with Capacity.

def theoretical_loss(N, capacity, beta_jamming=0.02, alpha_learning=0.0005):
    """
    Minimalist physical heuristic formula:
    L(N) = 1 - (1 - e^(-a*N*Cap)) * (1 / (1 + b*N))
    """
    total_capacity = N * capacity
    # Gain: Exponential saturation
    ideal_performance = 1.0 - np.exp(-alpha_learning * total_capacity)
    # Cost: Linear jamming decay
    efficiency = 1.0 / (1.0 + beta_jamming * N)
    
    final_loss = 1.0 - (ideal_performance * efficiency)
    return final_loss

def plot_theory_publication_ready():
    # 1. Data Preparation
    n_values = np.logspace(np.log10(2), np.log10(200), 200) 
    capacities = [40, 80, 160]
    
    # Professional color scheme
    colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
    linestyles = ['-.', '-', '--'] 
    
    # 2. Create Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    optima_points = []
    
    for i, cap in enumerate(capacities):
        # Calculate
        losses = theoretical_loss(n_values, capacity=cap)
        
        # Find optimum
        min_idx = np.argmin(losses)
        min_n = n_values[min_idx]
        min_loss = losses[min_idx]
        optima_points.append((min_n, min_loss))
        
        # Plot curve
        label = f'Capacity = {cap} (Opt $N^* \\approx {min_n:.1f}$)'
        ax.plot(n_values, losses, color=colors[i], linestyle=linestyles[i], 
                linewidth=2.5, label=label, alpha=0.9)
        
        # Mark optimum (Star)
        ax.scatter(min_n, min_loss, color=colors[i], s=150, marker='*', 
                   edgecolors='black', zorder=10)
        
        # Vertical line
        ax.vlines(min_n, 0, min_loss, color=colors[i], linestyle=':', alpha=0.4)

    # 3. Add Trend Arrow (From Blue Star to Green Star)
    start_n, start_l = optima_points[0] # Cap 40
    end_n, end_l = optima_points[-1]    # Cap 160
    
    ax.annotate("", 
                xy=(end_n, end_l + 0.02), 
                xytext=(start_n, start_l + 0.02),
                arrowprops=dict(arrowstyle="->", color="black", lw=2, 
                                connectionstyle="arc3,rad=-0.2"))
    
    ax.text((start_n + end_n)/2, start_l + 0.05, 
            "Higher Capacity $\\rightarrow$ Smaller $N^*$", 
            ha='center', fontsize=10, fontweight='bold')

    # 4. Layout & Aesthetics
    ax.set_xscale('log')
    
    # Auto-adjust Y-axis limits
    y_min = min(p[1] for p in optima_points)
    # Calculate max loss at boundaries to ensure curve fits
    y_max_check = max(theoretical_loss(n_values[0], 40), theoretical_loss(n_values[-1], 40))
    ax.set_ylim(y_min - 0.05, min(1.0, y_max_check + 0.05)) 
    
    ax.set_xlabel('Number of Particles ($N$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Theoretical Loss', fontsize=12, fontweight='bold')
    ax.set_title('Analytical Theory: Optimal $N^*$ shifts with Particle Capacity', fontsize=14, pad=15)
    
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    ax.legend(fontsize=11, loc='upper center', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('Figure_theory_v20.png')
    plt.show()
    
    print("Theoretical Optima Summary:")
    for i, cap in enumerate(capacities):
        print(f"Capacity {cap:3d}: N* = {optima_points[i][0]:.2f}")

if __name__ == "__main__":
    plot_theory_publication_ready()
