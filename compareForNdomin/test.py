#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : test.py
@Date  : 2019/03/16
@Desc  : 
"""
from MOEA.ndomin.NDSetDelete import NDSetDelete
from MOEA.ndomin.NDSetRankDelete import NDSetRankDelete
from MOEA.ndomin.NDSetSwk import NDSetSwk
from MOEA.ndomin.NDSetRankSwk import NDSetRankSwk
from MOEA.ndomin.NDSetRankDelete import NDSetRankDelete
from MOEA.ndomin.NDSetRankSBC_SS import NDSetRankSBC_SS
from MOEA.ndomin.NDSetRankSBC_BS import NDSetRankSBC_BS
from MOEA.ndomin.NDSetRankENS_SS import NDSetRankENS_SS
from MOEA.ndomin.NDSetRankENS_BS import NDSetRankENS_BS
from MOEA.sample.LatinHyperCube import LatinHyperCube
from MOEA.sample.randomSample import randomSample
import numpy as np
import csv
from pylab import *
from matplotlib.pyplot import plot as plt

if __name__ == '__main__':
	size = 2000
	dimension = 3
	D = randomSample(size, dimension)
	D1 = LatinHyperCube(size, dimension, span=(1, 20))//1
	D1 = np.array([[ 1,15,14],
				 [ 4, 7, 8],
				 [ 5, 2,10],
				 [ 7, 9, 1],
				 [10, 3, 6],
				 [10,18,15],
				 [14,11, 7],
				 [16,17,19],
				 [17, 6,18],
				 [19,12, 4]])
	NDSetDeleteS = time.clock()
	NDSetS = NDSetRankENS_BS(D)
	NDSetDeleteE = time.clock()
	NDSetDeleteT = NDSetDeleteE - NDSetDeleteS
	print(len(NDSetS))
	print('NDSetRankENS_BS：', NDSetDeleteT)
	# print(NDSetS[0])

	NDSetSwkS = time.clock()
	NDSetS = NDSetRankENS_SS(D)
	NDSetSwkE = time.clock()
	NDSetSwkT = NDSetSwkE - NDSetSwkS

	print('NDSetRankENS_SS：', NDSetSwkT)
	# print(NDSetS[0])

	NDSetSwkS = time.clock()
	NDSetS = NDSetRankSBC_SS(D)
	NDSetSwkE = time.clock()
	NDSetSwkT = NDSetSwkE - NDSetSwkS

	print('NDSetRankSBC_BS：', NDSetSwkT)
	# print(NDSetS[0])

	NDSetSwkS = time.clock()
	NDSetS = NDSetRankDelete(D)
	NDSetSwkE = time.clock()
	NDSetSwkT = NDSetSwkE - NDSetSwkS

	print('NDSetRankDelete：', NDSetSwkT)
	# print(NDSetS[0])

	NDSetSwkS = time.clock()
	NDSetS = NDSetRankSwk(D)
	NDSetSwkE = time.clock()
	NDSetSwkT = NDSetSwkE - NDSetSwkS

	print('NDSetRankSwk   ：', NDSetSwkT)
	# print(NDSetS[0])

