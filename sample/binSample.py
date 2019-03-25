#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : binSample.py
@Date  : 2019/01/13
@Desc  : 
"""
import numpy as np
import random
def binSample(size, binLength):
	population = np.empty((size, binLength))
	for i in range(size):
		chromosome = np.empty(binLength)
		for j in range(binLength):
			chromosome[j] = random.randint(0, 1)
		population[i] = chromosome
	return population

if __name__ == '__main__':
	p = binSample(10, 10)
	print(p)