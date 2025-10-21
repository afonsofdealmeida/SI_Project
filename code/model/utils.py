from utils.utils import *


# ---------------------------
# Gaussian Membership Function
# ---------------------------
class GaussianMF(nn.Module):
    def __init__(self, centers, sigmas, agg_prob):
        super().__init__()
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.sigmas = nn.Parameter(torch.tensor(sigmas, dtype=torch.float32))
        self.agg_prob=agg_prob

    def forward(self, x):
        if not isinstance(self.centers, torch.Tensor): 
             raise TypeError(f"self.centers is not a tensor! Found type: {type(self.centers)}")
        if not isinstance(self.sigmas, torch.Tensor):
             raise TypeError(f"self.sigmas is not a tensor! Found type: {type(self.sigmas)}")
        # Expand for broadcasting
        # x: (batch, 1, n_dims), centers: (1, n_rules, n_dims), sigmas: (1, n_rules, n_dims)
        diff = abs((x.unsqueeze(1) - self.centers.unsqueeze(0))/self.sigmas.unsqueeze(0)) #(batch, n_rules, n_dims)

        # Aggregation
        if self.agg_prob:
            dist = torch.norm(diff, dim=-1)  # (batch, n_rules) # probablistic intersection
        else:
            dist = torch.max(diff, dim=-1).values  # (batch, n_rules) # min intersection (min instersection of normal funtion is the same as the max on dist)
        
        return torch.exp(-0.5 * dist ** 2)


# ---------------------------
# TSK Model
# ---------------------------
class TSK(nn.Module):
    def __init__(self, n_inputs, n_rules, centers, sigmas,agg_prob=False):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Antecedents (Gaussian MFs)
        
        self.mfs=GaussianMF(centers, sigmas,agg_prob) 

        # Consequents (linear functions of inputs)
        # Each rule has coeffs for each input + bias
        self.consequents = nn.Parameter(
            torch.randn(n_inputs + 1,n_rules)
        )

    def forward(self, x):
        # x: (batch, n_inputs)
        batch_size = x.shape[0]
        
        # Compute membership values for each input feature
        # firing_strengths: (batch, n_rules)
        firing_strengths = self.mfs(x)
        
        # Normalize memberships
        # norm_fs: (batch, n_rules)
        norm_fs = firing_strengths / (firing_strengths.sum(dim=1, keepdim=True) + 1e-9)

        # Consequent output (linear model per rule)
        x_aug = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # add bias

        rule_outputs = torch.einsum("br,rk->bk", x_aug, self.consequents)  # (batch, rules)
        # Weighted sum
        output = torch.sum(norm_fs * rule_outputs, dim=1, keepdim=True)

        return output, norm_fs, rule_outputs
    

# ---------------------------
# Least Squares Solver for Consequents (TSK)
# ---------------------------
def train_ls(model, X, y):
    with torch.no_grad():
        _, norm_fs, _ = model(X)

        # Design matrix for LS: combine normalized firing strengths with input
        X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
        
        Phi = torch.einsum("br,bi->bri", X_aug, norm_fs).reshape(X.shape[0], -1)
        
        # Solve LS: consequents = (Phi^T Phi)^-1 Phi^T y
        
        theta= torch.linalg.lstsq(Phi, y).solution
    
        
        model.consequents.data = theta.reshape(model.consequents.shape)


# ---------------------------
# Gradient Descent Training 
# ---------------------------
def train_gd(model, X, y, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred, _, _ = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()


# ---------------------------
# Hybrid Training (Classic ANFIS)
# ---------------------------
def train_hybrid_anfis(model, X, y, max_iters=20, gd_epochs=20, lr=1e-3):
    train_ls(model, X, y)
    for _ in range(max_iters):
        # Step A: GD on antecedents (freeze consequents)
        model.consequents.requires_grad = False
        train_gd(model, X, y, epochs=gd_epochs, lr=lr)

        # Step B: LS on consequents (freeze antecedents)
        model.consequents.requires_grad = True
        model.mfs.requires_grad = False
        train_ls(model, X, y)

        # Re-enable antecedents
        model.mfs.requires_grad = True

def train_anfis(model, Xtr, ytr):

    # Ensure ytr is a column vector
    if len(ytr.shape) == 1:
        ytr = ytr.reshape(-1, 1)

    # Call your hybrid ANFIS training routine
    train_hybrid_anfis(model, Xtr, ytr)
    
    return model


def fold_accuracy(model, Xval, yval):
    """
    Compute accuracy of TSK model on a validation set.
    """
    y_pred, _, _ = model(Xval)
    y_pred_labels = (y_pred.detach().numpy() > 0.5).astype(int)
    return accuracy_score(yval.detach().numpy(), y_pred_labels)


def build_tsk_model(Xtr, ytr, n_clusters, centers, sigmas, TSK):
    """
    Build TSK model and convert data to torch tensors.
    
    Parameters:
    - Xtr, ytr: training data (NumPy arrays)
    - n_clusters: number of TSK rules/clusters
    - centers: cluster centers from Fuzzy C-means (n_clusters x n_features)
    - sigmas: cluster spreads (n_clusters x n_features)
    - TSK: TSK model class
    
    Returns:
    - model: TSK model instance
    - Xtr_tensor, ytr_tensor, Xte_tensor, yte_tensor: torch tensors
    """

    # Build TSK model (exclude last column if it's the target)
    model = TSK(
        n_inputs=Xtr.shape[1],
        n_rules=n_clusters,
        centers=centers[:, :-1],
        sigmas=sigmas[:, :-1]
    )

    # Convert to torch tensors
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)

    return model, Xtr, ytr

def compute_sigmas(Xexp, centers, u, m=2.0):
    """
    Compute cluster spreads (sigmas) for Fuzzy C-means clusters.
    Xexp, centers, u are expected to be NumPy arrays.
    """
    n_clusters = centers.shape[0]
    sigmas = []
    
    # Ensure Xexp is a NumPy array (it is, due to the fix in CV function)
    if not isinstance(Xexp, np.ndarray):
         Xexp = Xexp.cpu().numpy()

    for j in range(n_clusters):
        u_j = u[j, :] ** m
        # Xexp: (samples, features), centers[j]: (features,)
        # np.average operates correctly on NumPy arrays
        var_j = np.average((Xexp - centers[j])**2, axis=0, weights=u_j)
        sigma_j = np.sqrt(var_j)
        sigmas.append(sigma_j)
        
    return np.array(sigmas)
