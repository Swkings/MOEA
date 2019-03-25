#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : read.py
@Date  : 2018/12/04
@Desc  : 
"""
import numpy as np
import pandas as pd
dataSet = pd.read_csv("cityData20.csv", header=None)
X, Y, Z = dataSet[0].values, dataSet[1].values, dataSet[2].values
print(X, Y, Z)