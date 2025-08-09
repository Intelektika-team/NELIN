try:
    import numpy as np
    numpy = True
except:
    print("Please install numpy")
    numpy = False

try:import cupy as cp
except:pass
try:import matplotlib as plot
except:pass

if numpy:
    from nelin.other import fail_safe, time_profile
    from nelin.main import version as NELIN_VER
    from nelin.main import nelin_layer, nelin_rnnlayer, nelin_models, nelin_text, nlarr
    from nelin.models import nelin_fortest
    from nelin.text import nelin_chiper
    from nelin.functs import generate_xy, softmax, softsign, sigmoid, swish, relu, leaky_relu, tanh
    from nelin.main import greet
    import numpy as np
    adaptmod = nelin_models.adaptive_model
    array = nlarr
    ndarray = np.array
    