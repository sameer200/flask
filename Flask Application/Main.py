#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import itertools
from itertools import product
from sklearn.metrics import mean_squared_error
	
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import json
from statsmodels.tsa.seasonal import seasonal_decompose


def Process(grid_id):

	df = pd.DataFrame({})
	path = 'F:\Ovember'
	for i in range(1,8):
		df_new = pd.read_csv(path+'\sms-call-internet-mi-2013-11-0{}.csv'.format(i),parse_dates=['activity_date'])
		df = df.append(df_new)
		print("File " + str(i) + " added")


	df['activity_hour'] += 24*(df.activity_date.dt.day-1)


	# ## Series transformation

	df_grid = df[df['square_id']==grid_id]
	#df_grid.set_index('activity_hour', inplace=True) 
	df_grid.drop(['square_id', 'activity_date'], axis=1, inplace=True)
	#df_grid.to_csv('ts-grid-147.csv', index=False, encoding='utf-8')


	# # Split dataset into train and test
	train = df_grid[:125]
	test = df_grid[125:]


	train = train.set_index('activity_hour')    #Run this line once
	test = test.set_index('activity_hour')


	# Fit Arima model

	"""
	parameters_list - (p, q, P, Q) tuples
	p - associated with the auto-regressive aspect of the model
	d - integration order in ARIMA model (effects the amount of differencing to apply to a time series)
	D - seasonal integration order 
	s - length of season
	"""

	# AIC Scores
	# Akaike information criterion (AIC) (Akaike, 1974) 
	# is a fined technique based on in-sample fit 
	# to estimate the likelihood of a model to predict/estimate the future values. 
	# A good model is the one that has minimum AIC among all the other models.

	
	# setting initial values and some bounds for them
	ps = qs = Ps = Qs = range(0, 2)
	d = 1
	D = 1
	s = 24 # season length is 24

	# creating list with all the possible combinations of parameters
	parameters = product(ps, qs, Ps, Qs)
	parameters_list = list(parameters)

	best_aic = float("inf")
	best_param = parameters_list[0]

	for param in parameters_list:
		# since some combinations model fails to converge
		try:
		    model=sm.tsa.statespace.SARIMAX(train, order=(param[0], d, param[1]), 
					            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
		except:
		    continue
		aic = model.aic
		# saving best model, AIC and parameters
		if aic < best_aic:
		    best_aic = aic
		    best_param = param


	best_model = sm.tsa.statespace.SARIMAX(train,
		                        order=(best_param[0], d, best_param[1]),
		                        seasonal_order=(best_param[2], D, best_param[3], s),
		                        enforce_stationarity=False,
		                        enforce_invertibility=False).fit()


	#Predict list
	n = 41
	data = train.copy()
	data['arima_model'] = best_model.fittedvalues

	forecast = pd.DataFrame(best_model.predict(start=data.shape[0], end=data.shape[0]+n))

	forecast.to_csv('forecast1.csv', encoding='utf-8',header=False)
	colnames = ['activity_hour', 'total_activity']
	forecast = pd.read_csv('forecast1.csv', names=colnames, header=None)
	train.to_csv('train.csv', header=False , encoding='utf-8')
	test.to_csv('test.csv', header=False , encoding='utf-8')
	train = pd.read_csv('train.csv', names=colnames, header=None)
	test = pd.read_csv('test.csv', names=colnames, header=None)


	trainjsonpy = train.to_json("train.json", orient="records", date_format="epoch", double_precision=10,
				  force_ascii=True, date_unit="ms", default_handler=None)

	testjsonpy = test.to_json("test.json", orient="records", date_format="epoch", double_precision=10,
				 force_ascii=True, date_unit="ms", default_handler=None)

	forecastjsonpy = forecast.to_json("forecast.json", orient="records", date_format="epoch", double_precision=10,
					 force_ascii=True, date_unit="ms", default_handler=None)


	return







