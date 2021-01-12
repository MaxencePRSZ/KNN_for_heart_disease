from math import sqrt
from random import seed
from random import randrange
import pandas as pd


#Calculate the euclidean distance between two rows
def eucl_dist(vec1, vec2):
    distance = 0
    for i in range(len(vec1)-1):
        distance += (vec1[i] - vec2[i])**2
    distance = sqrt(distance)
    return distance

#Get all the K nearest neighbors
def get_n_neighbors(olds, new, k):
    distances = []
    neighbors = []
    for old in olds:
        dist = eucl_dist(old, new)
        distances.append((old, dist))
    distances.sort(key= lambda x : x[1])
    for distance in distances[:k]:
        neighbors.append(distance[0])
    return neighbors

# Predict the class according to the neighbors
def predict_classification(olds, new, k):
	neighbors = get_n_neighbors(olds, new, k)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# Split a dataset into k folds to make a cross validation
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted, corr_matrix):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
			if actual[i] == 1:
				corr_matrix[0][0]+=1
			else:
				corr_matrix[1][1]+=1
		else:
			if actual[i] == 1:
				corr_matrix[1][0]+=1
			else:
				corr_matrix[0][1]+=1
	return correct / float(len(actual)) * 100.0, corr_matrix


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	corr_matrices = list()
	for fold in folds:
		corr_matrix = [[0,0],[0,0]]
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy, corr_matrix = accuracy_metric(actual, predicted, corr_matrix)
		scores.append(accuracy)
		corr_matrices.append(corr_matrix)
	return scores, corr_matrices

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)
