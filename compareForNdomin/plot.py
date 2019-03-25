#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : plot.py
@Date  : 2019/03/16
@Desc  : 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MOEA.ndomin.NDSetSwk import NDSetSwk
data = pd.read_csv('./data/imgData.csv').values.astype(np.float)

# dimension-size
for dimension in range(2, 9):
	for N in range(250, 1001, 250):
		dIndex = np.where(data[:, 0]==dimension)
		dData = data[dIndex]
		sIndex = np.where(dData[:, 1]==N)
		sData = dData[sIndex]
		# alpha, FS, SBC = sData[:, 2], sData[:, 3], sData[:, 4]
		FS = sData[:, [2, 3]] * [-1, 1]
		SBC = sData[:, [2, 4]] * [-1, 1]
		NDSetIndex = NDSetSwk(SBC)
		FS = FS[NDSetIndex] * [-1, 1]
		SBC = SBC[NDSetIndex] * [-1, 1]
		fig, ax = plt.subplots()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		plt.plot(FS[:, 0], FS[:, 1], marker='+', color='red', label='FS')
		plt.plot(SBC[:, 0], SBC[:, 1], marker='x', color='blue', label='SBC')
		plt.legend(loc='upper right')
		text = '$dimension=' + str(dimension) + ', N=' + str(N) +'$'
		plt.text(SBC[-1][0], max(max(SBC[:, 1]), max(FS[:, 1])), text)
		plt.xlabel('$alpha$')
		plt.ylabel('$t$')
		path = './img/dimension-size/d' + str(dimension) + 's' + str(N) + '.eps'
		plt.savefig(path)

# dimension-alpha
# for dimension in range(2, 9):
# 	for i in range(1, 8):
# 		alpha = i/10
# 		dIndex = np.where(data[:, 0]==dimension)
# 		dData = data[dIndex]
# 		sData = np.array([item for item in dData if (alpha-0.05)<item[2]<(alpha+0.05)])
# 		if sData.shape[0] < 10:
# 			continue
# 		# alpha, FS, SBC = sData[:, 2], sData[:, 3], sData[:, 4]
# 		FS = sData[:, [1, 3]] * [-1, 1]
# 		SBC = sData[:, [1, 4]] * [-1, 1]
# 		NDSetIndex = NDSetSwk(SBC)
# 		FS = FS[NDSetIndex] * [-1, 1]
# 		SBC = SBC[NDSetIndex] * [-1, 1]
# 		fig, ax = plt.subplots()
# 		ax.spines['top'].set_visible(False)
# 		ax.spines['right'].set_visible(False)
# 		plt.plot(FS[:, 0], FS[:, 1], marker='+', color='red', label='FS')
# 		plt.plot(SBC[:, 0], SBC[:, 1], marker='x', color='blue', label='SBC')
# 		plt.legend(loc='upper right')
# 		text = '$dimension=' + str(dimension) + ', alpha=' + str(alpha) +'$'
# 		plt.text(SBC[-1][0], max(max(SBC[:, 1]), max(FS[:, 1])), text)
# 		plt.xlabel('$N$')
# 		plt.ylabel('$t$')
# 		path = './img/dimension-alpha/d' + str(dimension) + 'a' + str(alpha) + '.eps'
# 		plt.savefig(path)

# alpha-size
for i in range(1, 8):
	for N in range(250, 1001, 250):
		alpha = i/10
		sIndex = np.where(data[:, 1]==N)
		sData = data[sIndex]
		aData = np.array([item for item in sData if (alpha-0.05)<item[2]<(alpha+0.05)])
		try:
			FS = aData[:, [0, 3]]
			SBC = aData[:, [0, 4]]
		except IndexError:
			continue
		# FS = aData[:, [0, 3]] * [-1, 1]
		# SBC = aData[:, [0, 4]] * [-1, 1]
		# NDSetIndex = NDSetSwk(SBC)
		# 		# FS = FS[NDSetIndex] * [-1, 1]
		# 		# SBC = SBC[NDSetIndex] * [-1, 1]
		fig, ax = plt.subplots()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		plt.scatter(FS[:, 0], FS[:, 1], marker='+', color='red', label='FS', s=10)
		plt.scatter(SBC[:, 0], SBC[:, 1], marker='x', color='blue', label='SBC', s=10)
		maxD = int(max(SBC[:, 0]))
		plt.xticks(range(2, maxD+1))
		plt.legend(loc='upper right')
		text = '$alpha=' + str(alpha) + ', size=' + str(N) +'$'
		plt.text(2, max(max(SBC[:, 1]), max(FS[:, 1])), text)
		plt.xlabel('$dimension$')
		plt.ylabel('$t$')
		path = './img/alpha-size/a' + str(alpha) + 's' + str(N) + '.png'
		plt.savefig(path)
