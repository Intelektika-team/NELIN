import nelin.main as n
import numpy as np

class nelin_fortest:
    """Nelin test models. 
    #### Clases:
        xor: base neural networks problem."""
    class xor:
        """Base problem for testing neural networks. 
        - That model - only for testing, because that has 2 millions parameters.
        """
        def xorload() -> n.nelin_models.adaptive_model:
            """
            Function for load nelin XOR model. This only for testing, model hes more than 2 millions parameters. 

            Returns:
                - adaptive_model: you can use that like a other adaptive models.
            """
            net = n.nelin_models
            net1 = net.adaptive_model([2, 1000, 700, 600, 900, 200, 900, 1,], ["relu", "sigmoid", "linear", "relu", "tanh", "relu", "sigmoid"])
            model = net1.load("nelin/models/xorproblem")
            return model
        def xordata():
            """Data for testing XOR test model."""
            return np.array([[0,1], [1, 0], [0, 0], [1, 1]])
        def xortrain(save:bool=True):
            """# now is not working"""
            net = n.nelin_models
            net1 = net.adaptive_model([2, 1000, 700, 900, 900, 200, 900, 1], ["relu", "sigmoid", "linear", "relu", "tanh", "relu", "sigmoid"])
            y = np.array([1, 1, 0, 0])
            net1.train(nelin_fortest.xor.xordata(), y, 50000, 0.005)
            if save: net1.save("nelin/models/xorproblem")
        def xorchat():
            """Start chat with XOR test model.
            """
            xor = nelin_fortest.xor.xorload()
            while True:
                inp = input("\nYou: ").split()
                final_inp = []
                for i in inp:
                    try:final_inp.append(int(i))
                    except:pass
                try:outp = xor.predict(np.array([final_inp]))[-1]
                except Exception as e: print("= = Error in xor, ", e)
                try: print("= ", outp) 
                except: pass
    class notlnear:
        """Not linear problemfor testing neural networks.
        """
        def notlnearload() -> n.nelin_models.adaptive_model:
            """
            Function for load nelin notlinear model.

            Returns:
                - adaptive_model: you can use that like a other adaptive models.
            """
            net = n.nelin_models
            net1 = n.nelin_models.adaptive_model(
                [2, 640, 320, 2600, 1],
                ["relu", "linear", "sigmoid", "leaky_relu"],
                debug=[False, 100],
                leakyrelu=0.0001
            )
            model = net1.load("nelin/models/notlinear")
            return model
        def notlneartrain(save:bool=True):
            """# now is not working"""
            X = n.nlarr([[0, 10], [20, 0], [0, 30]])
            y = n.nlarr([1, 2, 4])
            X_np = X.numpy()
            X_norm = (X_np - np.mean(X_np, axis=0)) / np.std(X_np, axis=0)
            net = n.nelin_models.adaptive_model(
                [2, 640, 320, 2600, 1],
                ["relu", "linear", "sigmoid", "leaky_relu"],
                debug=[False, 100],
                leakyrelu=0.0001
            )
            net.train(X_norm, y.numpy(), epochs=9000, lr=0.0001, batch_size=3, earlystop=0.001, warning=1.2)
        def notlnearchat():
            """Start chat with notlnear test model.
            """
            nln = nelin_fortest.notlnear.notlnearload()
            while True:
                inp = input("\nYou: ").split()
                final_inp = []
                for i in inp:
                    try:final_inp.append(int(i))
                    except:pass
                try:outp = nln.predict(np.array([final_inp]))[-1]
                except Exception as e: print("= = Error in xor, ", e)
                try: print("= ", outp) 
                except: pass
    
    
    