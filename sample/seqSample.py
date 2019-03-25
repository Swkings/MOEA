#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : seqSample.py
@Date  : 2019/01/13
@Desc  : 
"""
import numpy as np
import random

def seqSample(size, seqLength):
	population = np.empty((size, seqLength))
	for i in range(size):
		chromosome = random.sample(range(seqLength), seqLength)
		population[i] = chromosome
	return population

if __name__ == '__main__':
	p = seqSample(10, 10)
	print(p)