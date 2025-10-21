from utils.utils import *

def evaluate_anfis(model, Xte, yte, threshold=0.5):
    """
    Evaluate trained ANFIS/TSK model and visualize prediction confidence.
    """

    # Predict
    y_pred, _, _ = model(Xte)
    y_pred = torch.sigmoid(y_pred)
    y_pred_np = y_pred.detach().numpy().flatten()
    y_pred_labels = (y_pred_np > threshold).astype(int)
    yte_np = yte.detach().numpy().flatten()

    # Metrics
    acc = accuracy_score(yte_np, y_pred_labels)
    precision = precision_score(yte_np, y_pred_labels)
    recall = recall_score(yte_np, y_pred_labels)
    f1 = f1_score(yte_np, y_pred_labels)
    cm = confusion_matrix(yte_np, y_pred_labels)

    # --- CONFIDENCE VISUALIZATION ---
    plt.figure(figsize=(10, 2))
    plt.axhline(0, color='gray', lw=0.5)
    
    for i, (score, true_label, pred_label) in enumerate(zip(y_pred_np, yte_np, y_pred_labels)):
        color = 'green' if true_label == pred_label else 'red'
        plt.scatter(score, 0, color=color, s=40, alpha=0.7)
    
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlim(0, 1)
    plt.yticks([])
    plt.xlabel("Predicted Confidence Score")
    plt.title("Prediction Confidence Visualization (Green = Correct, Red = Incorrect)")
    plt.legend()
    plt.show()

    # --- CONFUSION MATRIX ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Print metrics
    print(f"Final Test Accuracy: {acc:.4f}")
    print(f"Final Test Precision: {precision:.4f}")
    print(f"Final Test Recall: {recall:.4f}")
    print(f"Final Test F1-Score: {f1:.4f}")
