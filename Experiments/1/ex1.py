import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
print(parentparentdir)

import DataGenerator

df_train, df_val, df_test = DataGenerator.load_mame(dataframe=True)