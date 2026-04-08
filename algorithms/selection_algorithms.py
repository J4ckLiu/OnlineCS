import numpy as np
import pandas as pd 


def BH(q,pvals,  ntest):
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)
    
    
# ours
def online_BH(alpha, gamma, pval, n):
    """
    Input:
    alpha:   overall significance level, real number between 0 and 1.
    gamma:   weighting sequence, n-dim. vector of real numbers between 0 and 1 with sum <= 1.
    pvals:   p-values, n-dim. vector of real numbers between 0 and 1.
    n:       number of hypotheses, natural number.

    Output:
    rejects: n-dim. vector of rejection indicators (0 or 1).
    """

    gamma = np.asarray(gamma)
    pvals = np.asarray(pval.flatten())


    alpha_gamma_col = (alpha * gamma).reshape(-1, 1) # Shape (n, 1)
    ranks_row = np.arange(1, n + 1).reshape(1, -1) # Shape (1, n)
    thresholds_matrix = alpha_gamma_col * ranks_row
    pvals_col = pvals.reshape(-1, 1) 
    comparison_matrix = pvals_col <= thresholds_matrix
    rejection_counts = np.sum(comparison_matrix, axis=1)
    min_rej = n - rejection_counts + 1
    min_rej_col = min_rej.reshape(-1, 1)
    min_rej_comparison_matrix = min_rej_col <= ranks_row
    k_sums = np.sum(min_rej_comparison_matrix, axis=0)
    k_candidates_indices = np.where(k_sums >= np.arange(1, n + 1))[0]

    if k_candidates_indices.size > 0:
        k_star = max(k_candidates_indices[-1] + 1, 1) # Use [-1] for max as indices are sorted
    else:
        k_star = 1 

    rejects = (pvals <= alpha * gamma * k_star)
    rejected_indices = np.where(rejects)[0] 

    return rejected_indices
    
    
# competitor
def online_split(alpha, gamma,pval, n):

    rejects = pval <= alpha*gamma
    rejected_indices = np.where(rejects)[0]  # Extract indices where `rejects` is True

    return rejected_indices



