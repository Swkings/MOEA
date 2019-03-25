#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : main.py
@Date  : 2019/03/15
@Desc  : 
"""
from MOEA.ndomin.NDSetDelete import NDSetDelete
from MOEA.ndomin.NDSetRankDelete import NDSetRankDelete
from MOEA.ndomin.NDSetSwk import NDSetSwk
from MOEA.ndomin.NDSetRankSwk import NDSetRankSwk
from MOEA.sample.LatinHyperCube import LatinHyperCube
from MOEA.sample.randomSample import randomSample
import numpy as np
import csv
from pylab import *
from matplotlib.pyplot import plot as plt

if __name__ == '__main__':
	file = open('./data/imgData1.csv', 'w', newline='')
	fileW = csv.writer(file)
	fileW.writerow(['dimension','size','alpha','NDSetDeleteT','NDSetSwkT'])
	for d in range(2, 9):
		dimension = d
		if dimension != 8:
			continue
		for i in range(950, 1001):
			size = i
			if size % 250 == 0:
				for j in range(250):
					D = randomSample(size, dimension)
					NDSetDeleteS = time.clock()
					NDSetD = NDSetDelete(D)
					NDSetDeleteE = time.clock()
					NDSetDeleteT = NDSetDeleteE - NDSetDeleteS
					alpha = len(NDSetD)/size

					NDSetSwkS = time.clock()
					NDSetS = NDSetSwk(D)
					NDSetSwkE = time.clock()
					NDSetSwkT = NDSetSwkE - NDSetSwkS

					result = [dimension, size, alpha, NDSetDeleteT, NDSetSwkT]
					fileW.writerow(result)
			else:
				for j in range(10):
					D = randomSample(size, dimension)
					NDSetDeleteS = time.clock()
					NDSetD = NDSetDelete(D)
					NDSetDeleteE = time.clock()
					NDSetDeleteT = NDSetDeleteE - NDSetDeleteS
					alpha = len(NDSetD) / size

					NDSetSwkS = time.clock()
					NDSetS = NDSetSwk(D)
					NDSetSwkE = time.clock()
					NDSetSwkT = NDSetSwkE - NDSetSwkS

					result = [dimension, size, alpha, NDSetDeleteT, NDSetSwkT]
					fileW.writerow(result)
