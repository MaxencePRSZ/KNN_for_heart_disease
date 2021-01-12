from knn import k_nearest_neighbors, evaluate_algorithm
from dataprep import dataprep, print_corr_matr
import pandas as pd
from random import seed
from random import randrange



if __name__ == "__main__":
    df = dataprep()
    # Test the kNN on the Iris Flowers dataset
    seed(2)
    n_folds = 10
    num_neighbors = 3
    scores, corr_matrices = evaluate_algorithm(df, k_nearest_neighbors, n_folds, num_neighbors)
    print_corr_matr(corr_matrices)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
