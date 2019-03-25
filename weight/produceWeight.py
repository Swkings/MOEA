#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : produceWeight.py
@Date  : 2019/01/12
@Desc  :
"""
import MOEA.weight.permutations as perm
import numpy as np

def produceWeight(dimension, stepLength):
	"""
	产生某维度，每个维度步长
	:param dimension:
	:param stepLength:
	:return:
	"""
	seq = np.concatenate((np.zeros(stepLength), np.ones(dimension-1)))
	seqAll = perm.permutations(seq)
	weights = []
	for i in range(seqAll.shape[0]):
		index = np.where(seqAll[i] > 0)[0]
		index += 1
		index1 = np.append(index, dimension+stepLength)
		index2 = np.insert(index, 0, 0)
		weight = ((index1 - index2) - 1) / stepLength
		weights.append(weight)
	return np.array(weights)

if __name__ == '__main__':
	weights = produceWeight(3, 20)
	print(weights)
