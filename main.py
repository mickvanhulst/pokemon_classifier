import pandas as pd
import math
import numpy as np
import operator
 
# Turn off annoying warning (Link: http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)
pd.options.mode.chained_assignment = None

# Import functions
import KNN as knn
import bayesian as bayes

def main():
	# prepare data
	#load pokemon dataset. 
	data = pd.read_csv("./Pokemon.csv")
	# Determine target
	target_col_name = 'Type 1'

	# Features 
	# I decided not to use 'Legendary' & 'Generation' because the variation is to low. 
	# Classes like 'Bug' do not have a legendary card which results in an error: 'Divide by zero error'.
	features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

	# Split test/training dataset
	msk = np.random.rand(len(data)) < 0.80
	train_data = data[msk]
	test_data = data[~msk]

	# Add column classify
	test_data['classify'] = test_data.index

	# Determine k = n^0.5
	k = int(math.pow(len(data.index), 0.5))

	test_data_knn, accuracy_knn = knn.KNN(test_data, train_data, features, target_col_name, k)
	test_data_bayes, accuracy_bayes = bayes.gaussian_naive_bayes(test_data, train_data, features, target_col_name)

	print('KNN accuracy: ' + str(accuracy_knn))
	print('Gaussian naive bayes accuracy: ' + str(accuracy_bayes))

if __name__ == '__main__':
    main()