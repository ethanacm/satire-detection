import numpy as np
import pickle
import math
import argparse
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree



def numpy_fix_data(arr):
    non_zero = np.not_equal(np.longfloat(arr),np.longfloat(0))
    too_small = np.logical_and(non_zero, np.less(np.fabs(arr), np.float(.000000000000000001)))
    fixed_smalls = np.log10(arr, where = too_small)
    too_big = np.greater(np.fabs(arr),np.longfloat(1000000000))
    fixed_bigs = np.log10(arr, where = too_big)
    fix1done =  np.where(too_big, fixed_bigs * np.float(1000), np.longfloat(arr))
    fully_cleaned_up_data = np.where(too_small, fixed_smalls * np.float(1000000), np.longfloat(fix1done))
    return fully_cleaned_up_data
