#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : crowdingDistance.py
@Date  : 2019/01/01
@Desc  : 
"""
import numpy as np

def crowdingDistance(populationY):
	populationSize = populationY.shape[0]
	objFuncNum = populationY.shape[1]
	distance = np.zeros((populationSize))
	for i in range(objFuncNum):
		index = np.lexsort([populationY[:, i]])
		for j in range(populationSize):
			if j == 0 or j == populationSize-1:
				distance[index[j]] += 1000000
			else:
				distance[index[j]] += (populationY[index[j+1]][i] - populationY[index[j-1]][i])
	return distance