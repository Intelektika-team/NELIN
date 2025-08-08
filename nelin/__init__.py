from main import nelin_layer, nelin_models, nelin_text, nlarr
from models import nelin_fortest
from text import nelin_chiper
from functs import generate_xy, softmax, softsign, sigmoid, swish, relu, leaky_relu, tanh
from main import greet
import numpy as np
import sys
import os
script_path = os.path.abspath(__file__)
sys.path.append(script_path)


adaptmod = nelin_models.adaptive_model
array = nlarr