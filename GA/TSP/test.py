#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : test.py
@Date  : 2018/11/30
@Desc  : 
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import gzip

def _read32(bytestream):
  dt = np.dtype(np.int).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

path = './tspData/att48.tsp.gz'
with gzip.open(path) as bytestream:
	data = bytestream.readlines()[6:-1]
	data = [item.split() for item in data]
	data = np.array(data, dtype=np.int)
	Node , X, Y = data[:, 0], data[:, 1], data[:, 2]
	print(Node)
	print(X)
	print(Y)
	# data = data.reshape(num_images, rows, cols, 1)
	# print(np.array(data[6:], dtype=np.int))
	# t = {item.split(':', 1) for item in data[0:5]}
	# print(t)