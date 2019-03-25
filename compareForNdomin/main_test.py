#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : main_test.py
@Date  : 2019/03/22
@Desc  : 
"""
from pylab import *
import numpy as np
import random
import copy
import csv

from MOEA.ndomin.NDSetRankFast import NDSetRankFast
from MOEA.ndomin.NDSetRankBanker import NDSetRankBanker
from MOEA.ndomin.NDSetRankArena import NDSetRankArena
from MOEA.ndomin.NDSetRankDelete import NDSetRankDelete
from MOEA.ndomin.NDSetRankSwk import NDSetRankSwk
from MOEA.ndomin.NDSetRankSBC_BS import NDSetRankSBC_BS
from MOEA.ndomin.NDSetRankENS_BS import NDSetRankENS_BS

def randData(size, dimension):
	data = np.empty((size, dimension))
	for i in range(size):
	    for j in range(dimension):
	        data[i][j] = random.random()
	for dim in range(dimension):
		data[:, dim] = np.sort(data[:, dim])
	return data

if __name__ == '__main__':
	file = open('./data/nobj.csv', 'w', newline='')
	fileW = csv.writer(file)
	fileW.writerow(['dimension', 'size', 'rNum', 'NDSetRankFastT', 'NDSetRankBankerT', 'NDSetRankArenaT', 'NDSetRankDeleteT', 'NDSetRankSwkT', 'NDSetRankSBCT', 'NDSetRankENST'])
	size = 1000
	for dimension in [2, 5, 10]:
		for r in range(1, 100):
			if r % 10 == 0:
				print('dimension:{0}, rNum:{1}'.format(dimension, r))
			c = random.sample(range(0, dimension), random.randint(1, dimension - 1))
			rNum = size // r
			P = randData(size, dimension)
			for i in range(size):
				P[i, c] -= (i % rNum)
			np.random.shuffle(P)

			# NDSetRankFast
			FS = time.clock()
			NDSetF = NDSetRankFast(P)
			FE = time.clock()
			FT = FE - FS

			# NDSetRankBanker
			BS = time.clock()
			NDSetF = NDSetRankBanker(P)
			BE = time.clock()
			BT = BE - BS

			# NDSetRankArena
			AS = time.clock()
			NDSetA = NDSetRankArena(P)
			AE = time.clock()
			AT = AE - AS

			# NDSetRankDelete
			DS = time.clock()
			NDSetD = NDSetRankDelete(P)
			DE = time.clock()
			DT = DE - DS

			# NDSetRankSwk
			SS = time.clock()
			NDSetF = NDSetRankSwk(P)
			SE = time.clock()
			ST = SE - SS

			# NDSetRankSBC
			CS = time.clock()
			NDSetC = NDSetRankSBC_BS(P)
			CE = time.clock()
			CT = CE - CS

			# NDSetRankENS
			ES = time.clock()
			NDSetE = NDSetRankENS_BS(P)
			EE = time.clock()
			ET = EE - ES

			result = [dimension, size, rNum, FT, BT, AT, DT, ST, CT, ET]
			fileW.writerow(result)
