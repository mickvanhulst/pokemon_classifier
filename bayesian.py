#import packages
import numpy as np
import pandas as pd
import math

# Turn off annoying warning (Link: http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)
pd.options.mode.chained_assignment = None

def calc_mean_var(train_data, features, target_col_name):
	class_dict = {}

	for class_name in train_data[target_col_name].unique():
		summary_dict = {}
		
		for feature in features:
			# Calculate mean/variance
			mean = train_data[feature][train_data[target_col_name] == class_name].mean()
			std = train_data[feature][train_data[target_col_name] == class_name].std()

			# Store mean at pos 0 and variance at pos 1
			summary_dict[feature] = [mean, std]

		class_dict[class_name] = summary_dict
	return class_dict

def calc_probability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def det_classes_gaus(class_dict, test_data, train_data, target_col_name, features):
	# Loop through test dataset
	for idx in test_data.index:
		# Init list of chances
		chances = {}
		# Loop through all classes
		for class_name in train_data[target_col_name].unique():
			# Loop through all features
			chance = 1
			for feature in features:
				# Determine index pos of feature
				#feature_pos = train_data.columns.get_loc(feature)
				mean = class_dict[class_name][feature][0]
				std = class_dict[class_name][feature][1]

				if(std == 0.0):
					test_data.loc[idx, feature]
				chance += np.log(calc_probability(test_data.loc[idx, feature], mean, std))

			# Append chance per class
			chances[class_name] = chance

		# Append class with highest chance
		test_data.loc[idx, 'classify'] = max(chances, key=chances.get)

	# determine accuracy
	len_classify_equals_target = len(test_data[test_data[target_col_name] == test_data['classify']].index) 
	len_test_data = len(test_data.index)
	accuracy = len_classify_equals_target / len_test_data
				
	return test_data, accuracy


def gaussian_naive_bayes(test_data, train_data, features, target_col_name):
	# Calculate mean & variances for all features
	class_dict = calc_mean_var(train_data, features, target_col_name)

	# Determine classes for test data
	test_data, accuracy = det_classes_gaus(class_dict, test_data, train_data, target_col_name, features)

	return test_data, accuracy
