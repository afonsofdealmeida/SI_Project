from utils.utils import *
from model.utils import *
from clustering.utils import compute_sigmas

def tsk_cross_validation(Xtr, ytr, TSK, train_hybrid_anfis,
                         m_values=None, n_clusters_values=None,
                         n_splits=4, random_state=42):
    """
    Perform K-fold cross-validation to tune fuzziness parameter 'm' and number of clusters
    for a TSK model using hybrid ANFIS training.

    Parameters:
    - Xtr, ytr: training data (NumPy arrays or torch tensors)
    - TSK: TSK model class
    - train_hybrid_anfis: function to train TSK/ANFIS
    - m_values: list/array of fuzziness parameter values
    - n_clusters_values: list/array of cluster counts
    - n_splits: number of folds for CV
    - random_state: seed

    Returns:
    - best_params: (best_m, best_n_clusters)
    - best_acc: CV accuracy of best parameters
    """

    # Ensure inputs are NumPy arrays
    if torch.is_tensor(Xtr):
        Xtr = Xtr.detach().cpu().numpy()
    if torch.is_tensor(ytr):
        ytr = ytr.detach().cpu().numpy()

    if m_values is None:
        m_values = np.arange(1.85, 2.15, 0.05)
    if n_clusters_values is None:
        n_clusters_values = np.arange(2, 6, 1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params = None
    best_acc = 0.0

    for m in m_values:
        for n_clusters in n_clusters_values:
            fold_accs = []

            for train_idx, val_idx in kf.split(Xtr):
                # Split data and convert to torch tensors for PyTorch model
                Xtrain_fold = torch.tensor(Xtr[train_idx], dtype=torch.float32)
                Xval_fold = torch.tensor(Xtr[val_idx], dtype=torch.float32)
                ytrain_fold = torch.tensor(ytr[train_idx], dtype=torch.float32).reshape(-1,1)
                yval_fold = torch.tensor(ytr[val_idx], dtype=torch.float32).reshape(-1,1)

                # --- Fuzzy C-means on training fold (NumPy) ---
                Xtrain_np_fcm = Xtrain_fold.detach().cpu().numpy().T  # (features x samples)
                centers, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                    data=Xtrain_np_fcm,
                    c=n_clusters,
                    m=m,
                    error=0.005,
                    maxiter=1000
                )

                # --- Compute sigmas ---
                sigmas = compute_sigmas(Xtrain_fold.detach().cpu().numpy(), centers, u, m)

                # --- Build TSK model ---
                # Convert centers and sigmas to torch tensors inside TSK
                model = TSK(
                    n_inputs=Xtrain_fold.shape[1],
                    n_rules=n_clusters,
                    centers=centers,
                    sigmas=sigmas
                )

                # --- Train TSK with hybrid ANFIS ---
                train_hybrid_anfis(model, Xtrain_fold, ytrain_fold)

                # --- Evaluate fold ---
                acc = fold_accuracy(model, Xval_fold, yval_fold)
                fold_accs.append(acc)

            avg_acc = np.mean(fold_accs)
            print(f"m={m:.2f}, n_clusters={n_clusters}, CV Accuracy={avg_acc:.4f}")

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = (m, n_clusters)

    print("\nBest parameters:", best_params, "with CV Accuracy:", best_acc)
    return best_params, best_acc



