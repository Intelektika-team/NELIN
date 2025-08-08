import sys 
import os
import time
script_path = os.path.abspath(__file__)
sys.path.append(f"{script_path}/nelin")
import nelin 
nelin.greet()
time.sleep(4)