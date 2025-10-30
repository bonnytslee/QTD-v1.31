import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

# ==================== Sobol Sensitivity Analysis ====================
# Parameters (example for education pilot)
n_samples = 1000
n_params = 6

# Define parameter bounds (example: education intervention effects)
bounds = np.array([
    [0.1, 0.8],  # P1: Teacher training effect
    [0.0, 0.5],  # P2: Material cost reduction
    [0.2, 1.0],  # P3: Student engagement multiplier
    [50, 200],   # P4: Sample size per school
    [0.3, 0.9],  # P5: Baseline test score
    [0.1, 0.6]   # P6: Dropout rate
])

# Generate Sobol sequence
sampler = qmc.Sobol(d=n_params, scramble=True)
sample = sampler.random(n=n_samples)
params = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

# Model: Simulated test score improvement (Y)
def education_model(P):
    teacher_effect = P[0]
    cost_saving = P[1]
    engagement = P[2]
    n = P[3]
    baseline = P[4]
    dropout = P[5]
    
    # Simulated outcome
    effect = (teacher_effect * 0.4 + 
              cost_saving * 0.2 + 
              engagement * 0.3) * (1 - dropout)
    noise = np.random.normal(0, 0.05, len(P[0])) if P.ndim > 1 else 0
    return baseline + effect * n/100 + noise

# Run model
Y = education_model(params.T)

# ==================== First-order & Total-order Indices ====================
def sobol_indices(Y, n_params, n_samples):
    # Reshape
    N = n_samples
    Y = Y.reshape(-1)
    
    # Variance
    var_Y = np.var(Y)
    f0 = np.mean(Y)
    
    Si = np.zeros(n_params)
    STi = np.zeros(n_params)
    
    # For each parameter
    for i in range(n_params):
        # First-order: fix i, vary others
        A = params[:N, :]
        B = params[N:2*N, :]
        AB_i = B.copy()
        AB_i[:, i] = A[:, i]
        
        YA = education_model(A.T)
        YB = education_model(B.T)
        YAB = education_model(AB_i.T)
        
        # First-order index
        Si[i] = np.mean(YB * (YAB - YA)) / var_Y
        
        # Total-order index
        STi[i] = 1 - np.mean(YA * (YAB - YB)) / var_Y
    
    return Si, STi

# Calculate indices
Si, STi = sobol_indices(Y, n_params, n_samples//2)

# ==================== Plot Results ====================
plt.figure(figsize=(10, 6))
x = np.arange(n_params)
width = 0.35

plt.bar(x - width/2, Si, width, label='First-order (Sᵢ)', color='skyblue')
plt.bar(x + width/2, STi, width, label='Total-order (Sₜᵢ)', color='salmon')

plt.xlabel('Parameters')
plt.ylabel('Sobol Indices')
plt.title('Sobol Sensitivity Analysis - Education Pilot')
plt.xticks(x, [f'P{i+1}' for i in range(n_params)])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sobol_results.png', dpi=150)
plt.show()

# Print results
print("=== Sobol Sensitivity Results ===")
for i in range(n_params):
    print(f"P{i+1}: S1 = {Si[i]:.3f}, ST = {STi[i]:.3f}")

print(f"\nTotal variance explained: {np.sum(Si):.3f}")
