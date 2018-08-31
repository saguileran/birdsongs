#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:20:37 2018

@author: juan
"""

import pandas as pd
import os

# %%
base_path = '/home/juan/Documentos/Musculo/Codigo canarios/'
files = ['ff_SCI-all_2', 'ff_SCI-2018-06-15.17.30.25']
out_file = 'ff_SCI-all_3'
frames = [pd.read_csv('{}{}'.format(base_path, x), index_col=0) for x in files]
result = pd.concat(frames)
result.to_csv('{}{}'.format(base_path, out_file))
