#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : PseudoMonteCarlo.py
@Date  : 2019/01/11
@Desc  : 伪蒙特卡洛采样方法
"""
"""
1、将每个维度分成若干均等部分（箱子），在每个箱子中产生若干个数
"""
import numpy as np
import random
from functools import reduce
def PseudoMonteCarlo(box, boxINum=1, span=None):
	"""
	伪蒙特卡洛采样方法
	:param box: [a1, a2, ...ai]， 划分区间，每个维度的数代表划分的层数
	:param span: 每一维度的取值范围
	:param boxINum: 每一箱子产生数的个数
	:return:
	"""
	if type(box) is list:
		box = np.array(box)
	dimension = box.shape[0]
	layerLength = 1 / box  # 每个维度每层的区间长度
	boxCombine, boxNum = _combine(box)
	# 每个箱子的取值区间
	boxSpan = boxCombine*layerLength
	sample = np.zeros((boxNum*boxINum, dimension))
	for i in range(boxNum):
		for j in range(boxINum):
			sample[i*boxINum+j] = np.random.uniform(0, 1, dimension)
	if span is None:
		span = (np.zeros(dimension), np.ones(dimension))
	sample = sample * (span[1] - span[0]) + span[0]
	return sample


def _combine(box):
	"""
	产生组合数, 维度为box的维度，每个维度的数小于ai
	:param box: [a1, a2, ...ai]
	:return: boxIndex, boxNum
	"""
	boxNum = reduce(lambda x1, x2 : x1*x2, box)
	dimension = box.shape[0]
	boxIndex = np.zeros((boxNum, dimension))
	for boxI in range(boxNum):
		tempNum = boxI
		for i in range(dimension):
			boxIndex[boxI][i] = tempNum % box[i]
			tempNum //= box[i]
	return boxIndex, boxNum
if __name__ == '__main__':
	test = PseudoMonteCarlo([20, 20], 1)

	import matplotlib.pyplot as plt
	plt.scatter(test[:, 0], test[:, 1])
	plt.show()