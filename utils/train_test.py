import numpy as np

def train_test_split(R, test_ratio=0.2, seed=42):
    """
    Splits the ratings matrix R into training and testing sets.
    
    Args:
        R (np.ndarray): The full ratings matrix.
        test_ratio (float): The proportion of ratings to move to the test set.
        seed (int): Random seed for reproducibility.
        
    Returns:
        R_train (np.ndarray): Training matrix with some ratings masked (set to 0).
        R_test (np.ndarray): Test matrix containing only the masked ratings.
    """
    np.random.seed(seed)
    R_train = R.copy()
    R_test = np.zeros(R.shape)
    
    # Indices of non-zero ratings
    nonzero_indices = np.argwhere(R > 0)
    n_total = len(nonzero_indices)
    n_test = int(n_total * test_ratio)
    
    # Randomly select indices for the test set
    shuffled_idx = np.random.permutation(n_total)
    test_idx = shuffled_idx[:n_test]
    
    # Move selected ratings from train to test
    for idx in test_idx:
        r, c = nonzero_indices[idx]
        R_test[r, c] = R[r, c]
        R_train[r, c] = 0
        
    return R_train, R_test
