import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from datetime import date
import os

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scipy
import sympy

from scipy.special import gamma as Г

from scipy.optimize import minimize

import time
import tqdm as tqdm

import sympy

# Глобальные параметры
EPS = 0.00001  # точность find_value_robust
RAYS = 20  # количество лучей в сетке
NUM = 3  # количество точек на луче в сетке
SIZE = 20  # количество точек по которым смотрится максимум шума
SAMPLE_SIZE = 1000  # размер выборок при проверке на однородность
xmin, xmax, ymin, ymax = -100, 400, -100, 1200  # пределы для рисования контура

