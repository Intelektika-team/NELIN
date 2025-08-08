try:
    import numpy as np
    numpy = True
except:
    print("Please install numpy")
    numpy = False

from nelin.main import nelin_layer, nelin_models, nelin_text, nlarr
from nelin.models import nelin_fortest
from nelin.text import nelin_chiper
from nelin.functs import generate_xy, softmax, softsign, sigmoid, swish, relu, leaky_relu, tanh
from nelin.main import greet
import numpy as np
adaptmod = nelin_models.adaptive_model
array = nlarr