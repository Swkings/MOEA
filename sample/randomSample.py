#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : randomSample.py
@Date  : 2019/03/16
@Desc  : 
"""
import numpy as np
import random
def randomSample(size, dimension=2):
	data = np.empty((size, dimension))
	for i in range(size):
		X = np.empty(dimension)
		for j in range(dimension):
			X[j] = random.random()
		data[i] = np.array(X)
	return data