#A program that uses machine learning to predict the stock market
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import os
import urllib.request, json
import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


