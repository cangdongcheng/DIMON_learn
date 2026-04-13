import torch
import numpy as np

data_path = "../DIMON_training_data_healthy.npz"
dataset = np.load(data_path)
theta = dataset['theta']   
pacing = dataset['pacing'] 
u_all = dataset['u_data']  
cobiveco = dataset['cobiveco'] 
anisotropy = dataset['ref_anisotropy'] 
x_coords = dataset['cartesian_coords'] 


import torch
import numpy as np

def final_unit_test(u_data, x_coords, anisotropy):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Physical Parameters
    vf, vs, vn = 640.0, 240.0, 240.0
    
    # 2. Prepare Data (Using Heart 0, Pacing Site 0)
    # Ensure u_target is not all zeros!
    u_raw = u_data[0, 0, :]
    print(f"Target U - Mean: {u_raw.mean():.2f}, Max: {u_raw.max():.2f}")
    
    u_target = torch.tensor(u_raw, dtype=torch.float).to(device).unsqueeze(1)
    x_tensor = torch.tensor(x_coords, dtype=torch.float).to(device)
    
    # Normalization is key for the proxy to "wake up"
    x_min, x_max = x_tensor.min(0)[0], x_tensor.max(0)[0]
    spatial_range = x_max - x_min
    x_norm = (x_tensor - x_min) / spatial_range
    
    # 3. Robust Proxy Model
    # Using ReLU for the hidden layers to prevent Tanh saturation
    proxy = torch.nn.Sequential(
        torch.nn.Linear(3, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 1)
    ).to(device)
    
    # 4. Aggressive Fitting
    optimizer = torch.optim.Adam(proxy.parameters(), lr=1e-3)
    for i in range(2000):
        optimizer.zero_grad()
        u_pred = proxy(x_norm)
        loss = torch.nn.MSELoss()(u_pred, u_target)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(f"Step {i} | Proxy MSE: {loss.item():.2f}")

    # 5. Physics Backcheck
    x_norm.requires_grad_(True)
    u_pred = proxy(x_norm)
    grad_u_norm = torch.autograd.grad(u_pred, x_norm, torch.ones_like(u_pred), create_graph=True)[0]
    
    # Chain rule correction to get back to microns/ms
    grad_u = grad_u_norm / spatial_range
    
    # Construct D
    ani = torch.tensor(anisotropy, dtype=torch.float).to(device)
    f, s, n = ani[:, 0:3], ani[:, 3:6], ani[:, 6:9]
    D = (vf**2)*torch.einsum('ni,nj->nij', f, f) + \
        (vs**2)*torch.einsum('ni,nj->nij', s, s) + \
        (vn**2)*torch.einsum('ni,nj->nij', n, n)
    
    LHS = torch.sqrt(torch.einsum('ni,nij,nj->n', grad_u, D, grad_u) + 1e-8)
    residual = torch.abs(LHS - 1.0)
    
    print("-" * 30)
    print(f"Mean Grad Magnitude: {torch.norm(grad_u, dim=1).mean().item():.8f}")
    print(f"Mean Physics Residual: {residual.mean().item():.6f}")
    print("-" * 30)

final_unit_test(u_all, x_coords, anisotropy)