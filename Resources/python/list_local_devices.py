# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:53:01 2019

@author: public
"""
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
