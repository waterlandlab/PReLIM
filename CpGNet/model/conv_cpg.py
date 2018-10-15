import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D
from keras.layers import Embedding, GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D,Flatten,Input,LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model

from CpG_Net import CpGNet
from CpG_Bin import Bin
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn.metrics import roc_curve, auc

import random
from sklearn.metrics import roc_curve, auc
%load_ext autoreload
%autoreload 2
%matplotlib inline