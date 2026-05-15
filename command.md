create 'utils/metrics.py' contain BCEDiceloss and fill hole function

def get_postprocessed_mask(pred_logits, threshold=0.5):
    """Converts Logits to Binary Mask and applies Hole Filling"""
    preds = torch.sigmoid(pred_logits)
    preds = (preds > threshold).float()

    # Scipy needs numpy arrays
    preds_np = preds.cpu().numpy()
    filled_np = np.zeros_like(preds_np)

    # Loop through batch and channels to fill holes
    for b in range(preds_np.shape[0]):
        for c in range(preds_np.shape[1]):
            filled_np[b, c] = binary_fill_holes(preds_np[b, c]).astype(np.float32)

    # Return it as a PyTorch tensor on the same device
    return torch.from_numpy(filled_np).to(pred_logits.device)