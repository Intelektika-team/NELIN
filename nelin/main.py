import numpy as np
from nelin.functs import *
import random as r
from nelin.other import *
# cupy ->
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# numba ->
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Если Numba не установлен, создаем пустой декоратор
    def jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]  # Просто возвращаем функцию без изменений
        else:
            def decorator(func):
                return func
            return decorator
    NUMBA_AVAILABLE = False





version = "0.4.2 alpha-2"
def greet():
    """
    Welcome to NELIN!
    """
    # NEuraL engINe - NELIN
    print(f"""
  ===========================================
          
  ███╗   ██╗ ███████╗ ██╗      ██╗ ███╗   ██╗
  ████╗  ██║ ██╔════╝ ██║      ██║ ████╗  ██║
  ██╔██╗ ██║ █████╗   ██║      ██║ ██╔██╗ ██║
  ██║╚██╗██║ ██╔══╝   ██║      ██║ ██║╚██╗██║
  ██║ ╚████║ ███████╗ ███████╗ ██║ ██║ ╚████║
  ╚═╝  ╚═══╝ ╚══════╝ ╚══════╝ ╚═╝ ╚═╝  ╚═══╝
          
  Just simple, just write. That for you.
  Ver.{version}  By Intelektika
  ===========================================
          """)
# MARK: ===============



class nlarr:
    """ Nelin Array.
    Unified array type for Nelin with automatic CPU/GPU memory management
    
    Features:
    - Automatic device selection (GPU if available)
    - Seamless conversion between CPU/GPU
    - Full numpy/cupy API compatibility
    - Memory optimization
    - Unified interface for computations
    
    Parameters:
    data: array-like, np.ndarray, cp.ndarray or other
    device: 'auto', 'cpu', or 'gpu' (if available)
    dtype: data type (e.g., np.float32)
    """
    @fail_safe()
    def __init__(self, data, device='auto', dtype=None):
        self._device = None
        self._array = None
        self._dtype = dtype
        
        # Determine target device
        if device == 'auto':
            self._device = 'gpu' if CUPY_AVAILABLE else 'cpu'
        elif device == 'gpu' and not CUPY_AVAILABLE:
            print("Warning: CuPy not available. Falling back to CPU.")
            self._device = 'cpu'
        else:
            self._device = device
            
        # Convert input to array
        self._array = self._convert_to_device(data, self._device)
    
    @fail_safe()
    def _convert_to_device(self, data, target_device):
        """Convert data to target device with type handling"""
        # Handle nlarr inputs
        if isinstance(data, nlarr):
            return data.to(target_device)._array
        
        # Convert from numpy/cupy
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            if target_device == 'cpu':
                return cp.asnumpy(data).astype(self._dtype) if self._dtype else cp.asnumpy(data)
            return data.astype(self._dtype) if self._dtype else data
            
        if isinstance(data, np.ndarray):
            if target_device == 'gpu' and CUPY_AVAILABLE:
                return cp.asarray(data).astype(self._dtype) if self._dtype else cp.asarray(data)
            return data.astype(self._dtype) if self._dtype else data
        
        # Convert from other types (lists, tuples, etc.)
        if target_device == 'gpu' and CUPY_AVAILABLE:
            return cp.array(data, dtype=self._dtype) if self._dtype else cp.array(data)
        return np.array(data, dtype=self._dtype) if self._dtype else np.array(data)
    
    @fail_safe()
    @property
    def device(self):
        """Current storage device ('cpu' or 'gpu')"""
        return self._device
    
    @fail_safe()
    @property
    def shape(self):
        return self._array.shape
    
    @fail_safe()
    @property
    def dtype(self):
        return self._array.dtype
    
    @fail_safe()
    @property
    def nbytes(self):
        return self._array.nbytes
    
    @fail_safe()
    def to(self, device):
        """Convert to specified device"""
        if device == self._device:
            return self
        return nlarr(self._array, device=device)
    
    @fail_safe()
    def cpu(self):
        """Convert to CPU numpy array"""
        return self.to('cpu')
    
    @fail_safe()
    def gpu(self):
        """Convert to GPU cupy array (if available)"""
        return self.to('gpu')
    
    @fail_safe()
    def numpy(self):
        """Return as numpy array (always CPU)"""
        if self._device == 'gpu' and CUPY_AVAILABLE:
            return cp.asnumpy(self._array)
        return self._array
    
    @fail_safe()
    def cupy(self):
        """Return as cupy array (if on GPU)"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        if self._device == 'cpu':
            return cp.asarray(self._array)
        return self._array
    
    @fail_safe()
    def __array__(self):
        """Numpy array interface"""
        return self.numpy()
    
    @fail_safe()
    def __cupy_array__(self):
        """CuPy array interface"""
        return self.cupy()
    
    @fail_safe()
    def __repr__(self):
        return f"nlarr(device={self.device}, shape={self.shape}, dtype={self.dtype})\n" + repr(self._array)
    
    @fail_safe()
    def __str__(self):
        return str(self._array)
    
    @fail_safe()
    def __getitem__(self, key):
        return nlarr(self._array[key], device=self.device)
    
    @fail_safe()
    def __setitem__(self, key, value):
        if isinstance(value, nlarr):
            self._array[key] = value.to(self.device)._array
        else:
            self._array[key] = value
    
    @fail_safe()
    def __len__(self):
        return len(self._array)
    
    @fail_safe()
    def __eq__(self, other):
        if isinstance(other, nlarr):
            return nlarr(self._array == other.to(self.device)._array, device=self.device)
        return nlarr(self._array == other, device=self.device)
    
    # Mathematical operations
    @fail_safe()
    def __add__(self, other):
        return self._apply_op(other, '__add__')
    
    @fail_safe()
    def __sub__(self, other):
        return self._apply_op(other, '__sub__')
    
    @fail_safe()
    def __mul__(self, other):
        return self._apply_op(other, '__mul__')
    
    @fail_safe()
    def __truediv__(self, other):
        return self._apply_op(other, '__truediv__')
    @fail_safe()
    def __matmul__(self, other):
        return self._apply_op(other, '__matmul__')
    
    @fail_safe()
    def _apply_op(self, other, op_name):
        """Apply operation with automatic device conversion"""
        if isinstance(other, nlarr):
            # Ensure both arrays are on same device
            if self.device != other.device:
                other = other.to(self.device)
            other = other._array
            
        op = getattr(self._array, op_name)
        result = op(other)
        return nlarr(result, device=self.device)
    
    # Special functions
    @fail_safe()
    def sum(self, axis=None, keepdims=False):
        return nlarr(self._array.sum(axis=axis, keepdims=keepdims), device=self.device)
    
    @fail_safe()
    def mean(self, axis=None, keepdims=False):
        return nlarr(self._array.mean(axis=axis, keepdims=keepdims), device=self.device)
    
    @fail_safe()
    def max(self, axis=None, keepdims=False):
        return nlarr(self._array.max(axis=axis, keepdims=keepdims), device=self.device)
    
    @fail_safe()
    def min(self, axis=None, keepdims=False):
        return nlarr(self._array.min(axis=axis, keepdims=keepdims), device=self.device)
    
    @fail_safe()
    def transpose(self, axes=None):
        return nlarr(self._array.transpose(axes), device=self.device)
    
    @fail_safe()
    def reshape(self, shape):
        return nlarr(self._array.reshape(shape), device=self.device)
    
    @fail_safe()
    def copy(self):
        return nlarr(self._array.copy(), device=self.device)
    
    # Memory optimization
    @fail_safe()
    def pin_memory(self):
        """Pin memory for faster GPU transfers (if available)"""
        if self.device == 'cpu' and CUPY_AVAILABLE:
            return cp.cuda.alloc_pinned_memory(self.nbytes)
        return self
    
    # Custom serialization
    @fail_safe()
    def to_bytes(self):
        """Serialize to bytes with device info"""
        if self.device == 'gpu':
            array = self.cpu()._array
        else:
            array = self._array
            
        header = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'device': self.device
        }
        return header, array.tobytes()
    
    @classmethod
    @fail_safe()
    def from_bytes(cls, header, data):
        """Deserialize from bytes"""
        shape = header['shape']
        dtype = np.dtype(header['dtype'])
        device = header['device']
        
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        return cls(array, device=device)
    
    # Factory methods
    @staticmethod
    @fail_safe()
    def zeros(shape, dtype=np.float32, device='auto'):
        if device == 'auto':
            device = 'gpu' if CUPY_AVAILABLE else 'cpu'
            
        if device == 'gpu' and CUPY_AVAILABLE:
            return nlarr(cp.zeros(shape, dtype=dtype), device=device)
        return nlarr(np.zeros(shape, dtype=dtype), device=device)
    
    @staticmethod
    @fail_safe()
    def ones(shape, dtype=np.float32, device='auto'):
        if device == 'auto':
            device = 'gpu' if CUPY_AVAILABLE else 'cpu'
            
        if device == 'gpu' and CUPY_AVAILABLE:
            return nlarr(cp.ones(shape, dtype=dtype), device=device)
        return nlarr(np.ones(shape, dtype=dtype), device=device)
    
    @staticmethod
    @fail_safe()
    def empty(shape, dtype=np.float32, device='auto'):
        if device == 'auto':
            device = 'gpu' if CUPY_AVAILABLE else 'cpu'
            
        if device == 'gpu' and CUPY_AVAILABLE:
            return nlarr(cp.empty(shape, dtype=dtype), device=device)
        return nlarr(np.empty(shape, dtype=dtype), device=device)
    
    @staticmethod
    @fail_safe()
    def arange(start, stop=None, step=1, dtype=None, device='auto'):
        if device == 'auto':
            device = 'gpu' if CUPY_AVAILABLE else 'cpu'
            
        if device == 'gpu' and CUPY_AVAILABLE:
            return nlarr(cp.arange(start, stop, step, dtype=dtype), device=device)
        return nlarr(np.arange(start, stop, step, dtype=dtype), device=device)
# MARK: ===============



class nelin_layer:
    """
    A basic neural network layer implementation for the Nelin library
    
    Attributes:
        size (int): Number of neurons in the layer
        name (str): Optional identifier for the layer
        weights (ndarray): Weight matrix of shape (size, size)
        epoch (int): Training iteration counter
    
    Methods:
        train: Update layer weights using learning rule
        save: Save weights to file
        load: Load weights from file
        do: Perform forward pass through layer
        step: Get current training epoch count
    """
    
    def __init__(self, size, name=None):
        """
        Initialize neural network layer
        
        Parameters:
            size (int): Number of neurons in the layer
            name (str): Optional identifier for the layer
        """
        self.size = size
        self.name = name
        self.weights = np.random.rand(size, size)  # Random weight initialization
        self.epoch = 0
    
    def train(self, inp:list, true=None, lr=0.1):
        """
        Train the layer using either supervised or unsupervised learning
        
        Parameters:
            inp (array-like): Input vector (size must match layer size)
            true (array-like): Target vector for supervised learning 
                               (optional, size must match layer size)
            lr (float): Learning rate (default: 0.1)
        
        Returns:
            array: Output after training (inp.dot(weights))
        """
        inp = np.array(inp)

        if  len(inp) != self.size:
            print(f"Error in {self.name}: input size does not match layer size")
            return
    
        if true is not None:
            true = np.array(true)
            if len(inp) != self.size:
                print(f"Error in {self.name}: true output size does not match layer size")
                return
    
            # Supervised learning: simple weight update rule (gradient descent)
            output = np.dot(self.weights, inp)
            error = true - output
            # Update weights
            self.weights += lr * np.outer(error, inp)
    
        else:
            # Unsupervised learning (Hebbian learning)
            output = np.dot(self.weights, inp)
            self.weights += lr * np.outer(output, inp)
            
        self.epoch += 1
        return inp.dot(self.weights)
    
    def save(self, name=None):
        """
        Save layer weights to file
        
        Parameters:
            name (str): Filename to save weights. If None, uses layer's name.
        """
        if name is None:
            if self.name is None:
                print("Error: No filename specified and layer has no name")
                return
            name = self.name
        
        # Создаем директорию, если не существует
        import os
        os.makedirs("nelin/layerweights", exist_ok=True)
        
        full_path = f"nelin/layerweights/{name}"
        np.savetxt(full_path, self.weights)
        print(f"Saved weights to {full_path}")

    def load(self, name=None):
        """
        Load layer weights from file
        
        Parameters:
            name (str): Filename to load weights from. Uses layer's name if None.
        """
        if name is None:
            if self.name is None:
                print("Error: No filename specified and layer has no name")
                return
            name = self.name
        
        full_path = f"nelin/layerweights/{name}"
        try:
            self.weights = np.loadtxt(full_path)
            print(f"Loaded weights from {full_path}")
        except Exception as e:
            print(f"Error loading weights from {full_path}: {e}")
    
    def do(self, inp):
        """
        Perform forward pass through the layer
        
        Parameters:
            inp (array-like): Input vector
            
        Returns:
            array: Output vector (weights.dot(inp))
        """
        return self.weights.dot(inp)
    
    def step(self):
        """
        Get current training iteration count
        
        Returns:
            int: Current epoch count
        """
        return self.epoch
class nelin_rnnlayer:
    """
    Nelin vanilla RNN layer with BPTT (Backprop Through Time).
    
    Features:
      - Batch-first input: (batch, seq_len, input_size)
      - Multi-layer stacking (num_layers)
      - Optional bidirectional mode
      - Activation: 'tanh' or 'relu'
      - Full forward + backward (BPTT) and simple SGD update in `train`
      - Save / load to .npz
      - Automatic NumPy / CuPy backend selection (uses global `np`, `cp`, `CUPY_AVAILABLE`)
      - fail_safe protected methods for robustness
    
    Usage:
        rnn = nelin_rnnlayer(input_size=10, hidden_size=32, num_layers=1, activation='tanh', bidirectional=False)
        out, h_n = rnn.forward(x)  # x shape: (N, T, input_size)
        d_x = rnn.train(x, d_out, lr=0.001)  # where d_out is gradient wrt output (N, T, hidden_out)
    
    Notes:
      - Output shape: (N, T, hidden_size * (2 if bidirectional else 1))
      - h_n: last hidden state shape: (num_layers * num_directions, N, hidden_size)
    """

    @fail_safe()
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 activation: str = 'tanh', bidirectional: bool = False, bias: bool = True, name: str = None):
        # Basic params
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = max(1, int(num_layers))
        self.activation = activation.lower()
        if self.activation not in ('tanh', 'relu'):
            raise ValueError("activation must be 'tanh' or 'relu'")
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        self.use_bias = bool(bias)
        self.name = name

        # backend: numpy or cupy
        self._xp = cp if (CUPY_AVAILABLE and cp is not None) else np

        # initialize weights for each layer and each direction
        # We'll store parameters in dictionaries keyed by (layer, direction)
        self.weights_ih = {}  # input -> hidden
        self.weights_hh = {}  # hidden -> hidden
        self.bias_ih = {}     # bias for input->hidden
        self.bias_hh = {}

        # Xavier/Glorot-ish initialization scale
        for layer in range(self.num_layers):
            layer_input = self.input_size if layer == 0 else self.hidden_size * self.num_directions
            for direction in range(self.num_directions):
                key = (layer, direction)
                fan_in = layer_input
                fan_out = self.hidden_size
                bound = (6.0 / (fan_in + fan_out)) ** 0.5
                w_ih = self._xp.random.uniform(-bound, bound, (self.hidden_size, layer_input)).astype(self._xp.float32)
                w_hh = self._xp.random.uniform(-bound, bound, (self.hidden_size, self.hidden_size)).astype(self._xp.float32)
                self.weights_ih[key] = w_ih
                self.weights_hh[key] = w_hh
                if self.use_bias:
                    self.bias_ih[key] = self._xp.zeros((self.hidden_size,), dtype=self._xp.float32)
                    self.bias_hh[key] = self._xp.zeros((self.hidden_size,), dtype=self._xp.float32)
                else:
                    self.bias_ih[key] = None
                    self.bias_hh[key] = None

        # bookkeeping
        self.epoch = 0
        self._cache = {}  # store intermediate states for backward
        self._last_input_shape = None  # (N, T, input_size)
        self._xp_name = 'cupy' if (CUPY_AVAILABLE and cp is not None) else 'numpy'

    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_xp(self, arr=None):
        """Return xp (numpy or cupy) depending on arr or configured backend."""
        if arr is None:
            return self._xp
        if CUPY_AVAILABLE and cp is not None:
            # detect cupy array
            if hasattr(cp, 'ndarray') and isinstance(arr, cp.ndarray):
                return cp
        return np

    def _activate(self, x):
        xp = self._xp
        if self.activation == 'tanh':
            return xp.tanh(x)
        else:
            # relu
            return xp.maximum(0, x)

    def _activate_derivative(self, x):
        xp = self._xp
        if self.activation == 'tanh':
            return 1.0 - xp.tanh(x) ** 2
        else:
            return (x > 0).astype(x.dtype)

    # ---------------------------
    # Forward
    # ---------------------------
    @fail_safe()
    def forward(self, x, h0=None):
        """
        Forward pass through stacked (and optionally bidirectional) RNN.
        Args:
            x: ndarray (N, T, input_size)
            h0: optional initial hidden state, shape (num_layers * num_directions, N, hidden_size)
        Returns:
            output: ndarray (N, T, hidden_size * num_directions)
            h_n: ndarray last hidden states (num_layers * num_directions, N, hidden_size)
        Caches internal tensors for backward.
        """
        xp = self._get_xp(x)
        x = xp.array(x)
        N, T, _ = x.shape
        self._last_input_shape = (N, T, self.input_size)

        # initialize h0 if not provided
        if h0 is None:
            h_t = xp.zeros((self.num_layers * self.num_directions, N, self.hidden_size), dtype=x.dtype)
        else:
            h_t = xp.array(h0)

        # caches
        caches = {}  # per layer: store inputs, preacts, hiddens per time step
        layer_input = x  # shape (N, T, layer_input_size)

        for layer in range(self.num_layers):
            # prepare buffers for this layer
            # We'll produce outputs of shape (N, T, hidden_size * num_directions)
            outputs_per_direction = []
            layer_cache = {'preacts': [], 'hiddens': [], 'inputs': layer_input}  # inputs to this layer for backward
            for direction in range(self.num_directions):
                # direction: 0 = forward, 1 = backward (if bidirectional)
                key = (layer, direction)
                w_ih = self.weights_ih[key]
                w_hh = self.weights_hh[key]
                b_ih = self.bias_ih[key] if self.use_bias else None
                b_hh = self.bias_hh[key] if self.use_bias else None

                hidden_prev = h_t[layer * self.num_directions + direction]  # shape (N, hidden_size)
                # we'll store preactivations and hiddens for each time step
                preacts = []
                hiddens = []

                if direction == 0:
                    time_range = range(T)
                else:
                    time_range = range(T - 1, -1, -1)

                # iterate through time
                for t in time_range:
                    x_t = layer_input[:, t, :]  # (N, layer_input)
                    # linear transforms: W_ih @ x_t.T + W_hh @ h_prev.T  -> we compute in batch-friendly manner
                    # compute batch matmul: (N, hidden) = x_t @ W_ih.T + hidden_prev @ W_hh.T + biases
                    lin = x_t.dot(w_ih.T) + hidden_prev.dot(w_hh.T)
                    if b_ih is not None:
                        lin = lin + b_ih.reshape(1, -1)
                    if b_hh is not None:
                        lin = lin + b_hh.reshape(1, -1)
                    preacts.append(lin)
                    h_new = self._activate(lin)
                    hiddens.append(h_new)
                    hidden_prev = h_new  # for next time step

                # If backward, we stored in reversed time order; flip to forward time order for convenience
                if direction == 1:
                    preacts = preacts[::-1]
                    hiddens = hiddens[::-1]

                # stack to (N, T, hidden)
                preacts_arr = xp.stack(preacts, axis=1)
                hiddens_arr = xp.stack(hiddens, axis=1)
                outputs_per_direction.append(hiddens_arr)
                layer_cache.setdefault('preacts_' + str(direction), preacts_arr)
                layer_cache.setdefault('hiddens_' + str(direction), hiddens_arr)

            # concatenate directions along hidden dim
            if self.num_directions == 1:
                layer_out = outputs_per_direction[0]  # (N, T, hidden)
            else:
                layer_out = xp.concatenate(outputs_per_direction, axis=2)  # (N, T, hidden*2)

            caches[layer] = layer_cache
            # next layer input is this layer_out
            layer_input = layer_out

        # final outputs
        output = layer_input  # (N, T, hidden * num_directions)

        # pack last hidden states h_n (num_layers * num_directions, N, hidden_size)
        # last hidden for each layer and direction is last time-step hidden (for forward) or first(for backward)
        h_n = xp.zeros((self.num_layers * self.num_directions, N, self.hidden_size), dtype=x.dtype)
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                layer_cache = caches[layer]
                hiddens = layer_cache['hiddens_' + str(direction)]  # (N, T, hidden)
                if direction == 0:
                    h_last = hiddens[:, -1, :]
                else:
                    h_last = hiddens[:, 0, :]
                h_n[layer * self.num_directions + direction] = h_last

        # save caches for backward
        self._cache['caches'] = caches
        self._cache['output'] = output
        self._cache['h_n'] = h_n
        return output, h_n

    # alias
    @fail_safe()
    def __call__(self, x, h0=None):
        return self.forward(x, h0)

    # ---------------------------
    # Backward (BPTT)
    # ---------------------------
    @fail_safe()
    def backward(self, d_output):
        """
        Backprop through time.
        Args:
            d_output: gradient w.r.t. layer output (N, T, hidden * num_directions)
        Returns:
            d_x: gradient wrt input x (N, T, input_size)
            grads: dictionary with parameter gradients (weights_ih, weights_hh, bias_ih, bias_hh)
        """
        xp = self._get_xp(d_output)
        d_output = xp.array(d_output)
        N, T, out_dim = d_output.shape
        assert (N, T, out_dim) == self._cache['output'].shape, "d_output shape mismatch with cached forward output"

        caches = self._cache.get('caches', None)
        if caches is None:
            raise RuntimeError("No forward caches available for backward")

        # prepare gradient containers
        grads_w_ih = {k: xp.zeros_like(v) for k, v in self.weights_ih.items()}
        grads_w_hh = {k: xp.zeros_like(v) for k, v in self.weights_hh.items()}
        grads_b_ih = {k: xp.zeros_like(v) if v is not None else None for k, v in self.bias_ih.items()}
        grads_b_hh = {k: xp.zeros_like(v) if v is not None else None for k, v in self.bias_hh.items()}

        # gradient w.r.t. layer inputs (to propagate to previous layer)
        d_layer_input = None  # will become (N, T, in_features_for_this_layer)

        # iterate layers in reverse to compute parameter grads and propagate d_input
        for layer in reversed(range(self.num_layers)):
            layer_cache = caches[layer]
            # layer inputs: (N, T, in_features)
            layer_inputs = layer_cache['inputs']
            # create per-direction grads of shape (N, T, hidden)
            grad_per_direction = []
            for direction in range(self.num_directions):
                # d_out for this direction shape (N, T, hidden)
                if self.num_directions == 1:
                    d_out_dir = d_output  # (N, T, hidden)
                else:
                    # split d_output along hidden-dim
                    hidden = self.hidden_size
                    d_out_dir = d_output[:, :, direction * hidden:(direction + 1) * hidden]

                # We'll do BPTT for this direction
                key = (layer, direction)
                w_ih = self.weights_ih[key]   # (hidden, input_dim)
                w_hh = self.weights_hh[key]   # (hidden, hidden)
                b_ih = self.bias_ih[key] if self.use_bias else None
                b_hh = self.bias_hh[key] if self.use_bias else None

                preacts = layer_cache['preacts_' + str(direction)]  # (N, T, hidden)
                hiddens = layer_cache['hiddens_' + str(direction)]  # (N, T, hidden)

                # prepare gradients containers
                d_w_ih = xp.zeros_like(w_ih)
                d_w_hh = xp.zeros_like(w_hh)
                d_b_ih = xp.zeros_like(b_ih) if self.use_bias else None
                d_b_hh = xp.zeros_like(b_hh) if self.use_bias else None

                # gradient wrt hidden state at next time (initialized with zeros)
                d_h_next = xp.zeros((N, self.hidden_size), dtype=d_out_dir.dtype)

                # choose time iteration order (reverse for forward direction, forward for backward dir since we stored reversed caches accordingly)
                if direction == 0:
                    time_idxs = range(T - 1, -1, -1)
                else:
                    time_idxs = range(0, T)  # because cached arrays have been flipped to forward order

                # compute per-timestep
                # we need previous hidden for each time step (h_prev)
                for t in time_idxs:
                    # gradient coming from output at this time for this direction
                    d_out_t = d_out_dir[:, t, :]  # (N, hidden) - gradient w.r.t. this direction's output
                    # total gradient on activation preact is d_out_t + d_h_next (propagated from t+1)
                    total_dh = d_out_t + d_h_next  # (N, hidden)
                    # derivative of activation
                    grad_act = self._activate_derivative(preacts[:, t, :])  # (N, hidden)
                    d_preact = total_dh * grad_act  # (N, hidden)

                    # input at this time to this layer
                    x_t = layer_inputs[:, t, :]  # (N, input_dim)
                    # previous hidden: for t==0 prev is zeros
                    if t == 0:
                        h_prev = self._get_xp(layer_inputs) * 0  # dummy zero array with proper dtype and shape (will not be used for W_hh at t==0)
                        # but we should really use zeros of shape (N, hidden)
                        h_prev = xp.zeros((N, self.hidden_size), dtype=d_preact.dtype)
                    else:
                        h_prev = hiddens[:, t - 1, :]

                    # accumulate gradients
                    # d_w_ih += d_preact.T @ x_t
                    d_w_ih += d_preact.T.dot(x_t)
                    # d_w_hh += d_preact.T @ h_prev
                    d_w_hh += d_preact.T.dot(h_prev)
                    if self.use_bias:
                        d_b_ih += d_preact.sum(axis=0)
                        d_b_hh += d_preact.sum(axis=0)

                    # propagate gradient to previous hidden: d_h_next = d_preact @ W_hh
                    d_h_next = d_preact.dot(w_hh)

                # finished time loop for this direction
                grads_w_ih[key] = grads_w_ih.get(key, 0) + d_w_ih
                grads_w_hh[key] = grads_w_hh.get(key, 0) + d_w_hh
                if self.use_bias:
                    grads_b_ih[key] = grads_b_ih.get(key, 0) + d_b_ih
                    grads_b_hh[key] = grads_b_hh.get(key, 0) + d_b_hh

                # gradient to propagate to layer input (from this direction) is d_preact @ W_ih (summed over timesteps)
                # compute per-time contribution and sum
                # d_x_dir[t] = d_preact[t] @ W_ih  (N, input_dim)
                d_x_dir = xp.zeros_like(layer_inputs)
                # reconstruct d_preact per timestep by repeating the BPTT (inefficient but straightforward)
                # We can recompute per-timestep d_preact by re-running BPTT forward of cached values (cheap compared to full autograd)
                d_h_next_2 = xp.zeros((N, self.hidden_size), dtype=d_output.dtype)
                if direction == 0:
                    time_idxs2 = range(T - 1, -1, -1)
                else:
                    time_idxs2 = range(0, T)
                for t in time_idxs2:
                    pre = preacts[:, t, :]
                    grad_act = self._activate_derivative(pre)
                    # d_out contribution at t:
                    if self.num_directions == 1:
                        d_out_t = d_output[:, t, :]
                    else:
                        # already split as d_out_dir earlier
                        d_out_t = d_out_dir[:, t, :]
                    total_dh = d_out_t + d_h_next_2
                    d_preact = total_dh * grad_act
                    x_t = layer_inputs[:, t, :]
                    # accumulate input gradient
                    d_x_dir[:, t, :] = d_preact.dot(w_ih)
                    # next
                    d_h_next_2 = d_preact.dot(w_hh)

                grad_per_direction.append(d_x_dir)

            # sum gradients from both directions to get d_layer_input (since layer input receives contributions)
            if self.num_directions == 1:
                d_layer_input = grad_per_direction[0]
            else:
                # for bidirectional, each direction produced grads wrt same layer input dims (since forward and backward both read same input)
                # if input to this layer had fewer dims (for higher layers), directions concatenated; need to add properly
                # For our forward we concatenated outputs along hidden dim for next layer input, but inputs to this layer are same for both directions.
                # So simply sum per-direction gradients
                d_layer_input = xp.sum(xp.stack(grad_per_direction, axis=0), axis=0)

            # accumulate param grads already stored in grads_w_ih / grads_w_hh / grads_b_ih / grads_b_hh

        # normalize grads by batch size
        batch = self._last_input_shape[0] if self._last_input_shape is not None else N
        for k in grads_w_ih:
            grads_w_ih[k] = grads_w_ih[k] / max(1, batch)
            grads_w_hh[k] = grads_w_hh[k] / max(1, batch)
            if self.use_bias and grads_b_ih[k] is not None:
                grads_b_ih[k] = grads_b_ih[k] / max(1, batch)
                grads_b_hh[k] = grads_b_hh[k] / max(1, batch)

        grads = {
            'w_ih': grads_w_ih,
            'w_hh': grads_w_hh,
            'b_ih': grads_b_ih,
            'b_hh': grads_b_hh
        }
        return d_layer_input, grads

    # ---------------------------
    # Train (one SGD step)
    # ---------------------------
    @fail_safe()
    def train(self, x, grad_output, lr=0.01):
        """
        One training step:
          - ensures forward cache exists (calls forward if needed)
          - performs backward to compute gradients
          - updates parameters with SGD (simple step)
        Args:
            x: input (N, T, input_size)
            grad_output: gradient w.r.t. output (N, T, hidden * num_directions)
            lr: learning rate
        Returns:
            d_x: gradient wrt input x (N, T, input_size)
        """
        xp = self._get_xp(x)
        # ensure forward cache
        if 'output' not in self._cache:
            _ = self.forward(x)

        d_x, grads = self.backward(grad_output)

        # apply SGD updates
        for k, g in grads['w_ih'].items():
            self.weights_ih[k] = self.weights_ih[k] - lr * g
        for k, g in grads['w_hh'].items():
            self.weights_hh[k] = self.weights_hh[k] - lr * g
        if self.use_bias:
            for k, g in grads['b_ih'].items():
                if g is not None:
                    self.bias_ih[k] = self.bias_ih[k] - lr * g
            for k, g in grads['b_hh'].items():
                if g is not None:
                    self.bias_hh[k] = self.bias_hh[k] - lr * g

        self.epoch += 1
        return d_x

    # ---------------------------
    # Save / Load
    # ---------------------------
    @fail_safe()
    def save(self, name=None):
        """
        Save layer parameters into nelin/layerweights/<name>.npz
        Stored fields:
          - all weights and biases converted to numpy arrays
          - config dict
        """
        import os
        os.makedirs("nelin/layerweights", exist_ok=True)
        if name is None:
            if self.name is None:
                raise ValueError("No filename specified and layer has no name")
            name = self.name
        full_path = f"nelin/layerweights/{name}.npz"

        # collect params
        to_save = {}
        for k, v in self.weights_ih.items():
            kk = f'w_ih_{k[0]}_{k[1]}'
            if CUPY_AVAILABLE and cp is not None and hasattr(cp, 'asnumpy') and isinstance(v, cp.ndarray):
                to_save[kk] = cp.asnumpy(v)
            else:
                to_save[kk] = v
        for k, v in self.weights_hh.items():
            kk = f'w_hh_{k[0]}_{k[1]}'
            if CUPY_AVAILABLE and cp is not None and hasattr(cp, 'asnumpy') and isinstance(v, cp.ndarray):
                to_save[kk] = cp.asnumpy(v)
            else:
                to_save[kk] = v
        if self.use_bias:
            for k, v in self.bias_ih.items():
                kk = f'b_ih_{k[0]}_{k[1]}'
                if CUPY_AVAILABLE and cp is not None and hasattr(cp, 'asnumpy') and isinstance(v, cp.ndarray):
                    to_save[kk] = cp.asnumpy(v)
                else:
                    to_save[kk] = v
            for k, v in self.bias_hh.items():
                kk = f'b_hh_{k[0]}_{k[1]}'
                if CUPY_AVAILABLE and cp is not None and hasattr(cp, 'asnumpy') and isinstance(v, cp.ndarray):
                    to_save[kk] = cp.asnumpy(v)
                else:
                    to_save[kk] = v

        config = {
            'input_size': int(self.input_size),
            'hidden_size': int(self.hidden_size),
            'num_layers': int(self.num_layers),
            'activation': str(self.activation),
            'bidirectional': bool(self.bidirectional),
            'use_bias': bool(self.use_bias)
        }
        to_save['config'] = np.array([config], dtype=object)
        np.savez(full_path, **to_save)
        print(f"Saved RNN layer to {full_path}")

    @classmethod
    @fail_safe()
    def load(cls, file_path):
        """
        Load saved RNN layer from .npz
        """
        if not file_path.endswith('.npz'):
            file_path_s = f"{file_path}.npz"
        else:
            file_path_s = file_path
        data = np.load(file_path_s, allow_pickle=True)
        config = data['config'][0].tolist()

        layer = cls(
            input_size=int(config['input_size']),
            hidden_size=int(config['hidden_size']),
            num_layers=int(config['num_layers']),
            activation=str(config['activation']),
            bidirectional=bool(config['bidirectional']),
            bias=bool(config['use_bias'])
        )

        # load weights
        for k in list(layer.weights_ih.keys()):
            keyname = f'w_ih_{k[0]}_{k[1]}'
            if keyname in data:
                layer.weights_ih[k] = data[keyname]
        for k in list(layer.weights_hh.keys()):
            keyname = f'w_hh_{k[0]}_{k[1]}'
            if keyname in data:
                layer.weights_hh[k] = data[keyname]
        if layer.use_bias:
            for k in list(layer.bias_ih.keys()):
                keyname = f'b_ih_{k[0]}_{k[1]}'
                if keyname in data:
                    layer.bias_ih[k] = data[keyname]
            for k in list(layer.bias_hh.keys()):
                keyname = f'b_hh_{k[0]}_{k[1]}'
                if keyname in data:
                    layer.bias_hh[k] = data[keyname]

        # convert to cupy arrays if backend set to cupy
        if CUPY_AVAILABLE and cp is not None and layer._xp is cp:
            for k in layer.weights_ih:
                layer.weights_ih[k] = cp.array(layer.weights_ih[k])
            for k in layer.weights_hh:
                layer.weights_hh[k] = cp.array(layer.weights_hh[k])
            if layer.use_bias:
                for k in layer.bias_ih:
                    if layer.bias_ih[k] is not None:
                        layer.bias_ih[k] = cp.array(layer.bias_ih[k])
                for k in layer.bias_hh:
                    if layer.bias_hh[k] is not None:
                        layer.bias_hh[k] = cp.array(layer.bias_hh[k])

        print(f"Loaded RNN layer from {file_path_s}")
        return layer

    @fail_safe()
    def step(self):
        """Return current epoch (training step) counter."""
        return self.epoch

# MARK: ===============



class nelin_models:
    """
    Collection of neural network models for the Nelin library
    
    Contains:
        adaptive_model: Flexible neural network with backpropagation
        simple_model: Simplified sequential model
    """
    class adaptive_model:
        """
        Flexible neural network model with configurable architecture
        
        Attributes:
            layers (list): Layer configurations
            activations (list): Activation functions for each layer
        
        Methods:
            forward: Perform forward pass
            backward: Perform backpropagation
            apply_activation: Apply activation function
            apply_activation_derivative: Compute activation derivative
            train: Train the model
            predict: Make predictions with trained model
        """
        
        def __init__(self, layer_sizes, activations=None, debug=[False, 10], leakyrelu=0.01,):
            """
            Initialize neural network model
            
            Parameters:
                layer_sizes (list): List of layer sizes [input, hidden1, ..., output]
                activations (list): Activation functions for each layer (except input)(sigmoid, relu, linear, tanh, softmax, softsign, swish, leaky_relu)
                debug (list): parameters [Debug_available:bool, epochsdebug:int]
            """
            self.debug = bool(debug[0])
            try: self.debugepoch = int(debug[1])
            except: self.debugepoch = 1000
            self.layers = []
            self.activations = activations or ['relu'] * (len(layer_sizes)-1)
            self.layer_sizes = layer_sizes  # Сохраняем архитектуру сети
            self.debug_settings = debug  # Сохраняем настройки отладки
            
            # Initialize weights and biases
            for i in range(len(layer_sizes)-1):
                # Xavier/Glorot weight initialization
                bound = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
                weights = np.random.uniform(-bound, bound, (layer_sizes[i+1], layer_sizes[i]))
                biases = nlarr.zeros((layer_sizes[i+1], 1))
                
                self.layers.append({
                    'weights': weights,
                    'biases': biases,
                    'activation': self.activations[i]
                })
            self.forreluleak = leakyrelu

        def forward(self, X):
            """
            Perform forward pass through network
            
            Parameters:
                X (ndarray): Input data (shape: features x samples)
            
            Returns:
                ndarray: Network output
            """
            self.activations_cache = [X]
            self.z_cache = []
            a = X
            for layer in self.layers:
                z = layer['weights'] @ a + layer['biases']
                a = self.apply_activation(z, layer['activation'], self.forreluleak)
                
                self.z_cache.append(z)
                self.activations_cache.append(a)
                
            return a

        def backward(self, X, y, lr=0.01):
            """
            Perform backpropagation to update weights
            
            Parameters:
                X (ndarray): Input data
                y (ndarray): Target values
                lr (float): Learning rate
            """
            # Forward pass if not already done
            if not hasattr(self, 'activations_cache'):
                self.forward(X)
            
            # Calculate output layer gradient
            grads = []
            output = self.activations_cache[-1]
            error = output - y
            
            # Backward pass
            for i in reversed(range(len(self.layers))):
                # Activation gradient
                activation = self.layers[i]['activation']
                z = self.z_cache[i]
                delta = error * self.apply_activation_derivative(z, activation, self.forreluleak)
                
                # Save gradient for weight update
                prev_activation = self.activations_cache[i]
                grads.append({
                    'weights': delta @ prev_activation.T,
                    'biases': delta
                })
                
                # Propagate error to previous layer
                if i > 0:
                    error = self.layers[i]['weights'].T @ delta

            # Update weights (in reverse gradient order)
            for i, grad in enumerate(reversed(grads)):
                self.layers[i]['weights'] -= lr * grad['weights']
                self.layers[i]['biases'] -= lr * grad['biases'].mean(axis=1, keepdims=True)

        def apply_activation(self, x, name, leaky = 0.01):
            """
            Apply specified activation function
            
            Parameters:
                x (ndarray): Input values
                name (str): Activation function name
            
            Returns:
                ndarray: Transformed values
            
            Raises:
                ValueError: Unknown activation function
            """
            if name == 'sigmoid':
                return 1 / (1 + np.exp(-x))
            elif name == 'relu':
                return np.maximum(0, x)
            elif name == 'tanh':
                return np.tanh(x)
            elif name == 'softmax':
                ex = np.exp(x - np.max(x))
                return ex / np.sum(ex, axis=0)
            elif name == 'linear':
                return x
            elif name == 'softsign':
                return softsign(x)
            elif name == 'swish':
                return swish(x)
            elif name == 'leaky_relu':
                return leaky_relu(x, leaky)
            else:
                raise ValueError(f"Unknown activation: {name}")

        def apply_activation_derivative(self, x, name, leaky=0.01):
            """
            Compute derivative of specified activation function
            
            Parameters:
                x (ndarray): Input values
                name (str): Activation function name
            
            Returns:
                ndarray: Activation derivatives
            
            Raises:
                ValueError: Unknown activation function
            """
            if name == 'sigmoid':
                s = 1 / (1 + np.exp(-x))
                return s * (1 - s)
            elif name == 'relu':
                return (x > 0).astype(float)
            elif name == 'tanh':
                return 1 - np.tanh(x)**2
            elif name == 'softmax':
                s = self.apply_activation(x, 'softmax')
                return s * (1 - s)  # Simplified version for MSE
            elif name == 'linear':
                return np.ones_like(x)
            elif name == 'softsign':
                return softsign(x)
            elif name == 'swish':
                return swish(x)
            elif name == 'leaky_relu':
                return leaky_relu(x, leaky)
            else:
                raise ValueError(f"Unknown activation: {name}")

        
        def train(self, X, y, epochs=100, lr=0.01, batch_size=32, earlystop=0, warning=0):
            """
            Train model using mini-batch gradient descent
            
            Parameters:
                X (ndarray): Training features (samples x features)
                y (ndarray): Training labels (samples x outputs)
                epochs (int): Number of training iterations, epochs can be infinite with epochs="inf"
                lr (float): Learning rate
                batch_size (int): Size of mini-batches
            """
            if epochs=="inf": epochs = 999999999 ** 9
            if earlystop != 0: self.earlystop=True
            else: self.earlystop=False
            self.earlycoif = earlystop
            for epoch in range(epochs+1):
                epoch_loss = 0
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i+batch_size].T
                    try:
                        y_batch = y[i:i+batch_size].reshape(-1, 1).T 
                    except:
                        y_batch = y+batch_size
                    
                    # Forward pass
                    output = self.forward(X_batch)
                    
                    # Backward pass
                    self.backward(X_batch, y_batch, lr)
                    
                    # Calculate loss
                    loss = np.mean((output - y_batch)**2)
                    epoch_loss += loss
                predict = nelin_models.adaptive_model.predict(self, X)
                if epoch % round(epochs/self.debugepoch) == 0 and self.debug:
                    print(f"Epoch {epoch}, Loss: {epoch_loss/len(X):.6f}")
                    #print(nd_error(predict, y), self.earlycoif)
                
                if (nd_error(predict, y)) >= (warning*1.5):
                    if self.debug: print("Warning_error stop in epoch: ", epoch )
                    break
                if self.earlystop and nd_error(predict, y) <= self.earlycoif:
                    if self.debug: print("Early stop in epoch: ", epoch )
                    break

        def predict(self, X):
            """
            Make predictions using trained model
            
            Parameters:
                X (ndarray): Input data (samples x features)
            
            Returns:
                ndarray: Model predictions
            """
            try:
                return self.forward(X.T)
            except:
                return self.forward(X)
            

        def save(self, file_path):
            """
            Сохраняет модель в файл формата .npz
            
            Parameters:
                file_path (str): Путь для сохранения файла
            """
            save_data = {
                'layer_sizes': np.array(self.layer_sizes),
                'activations': np.array(self.activations, dtype=object),
                'debug_settings': np.array(self.debug_settings, dtype=object)
            }
            
            # Добавляем веса и смещения для каждого слоя
            for i, layer in enumerate(self.layers):
                save_data[f'weights_{i}'] = layer['weights']
                save_data[f'biases_{i}'] = layer['biases']
            
            # Сохраняем все данные в один файл
            np.savez(file_path, **save_data)
            print(f"Model saved to {file_path}")

        @classmethod
        def load(cls, file_path):
            """
            Загружает модель из файла
            
            Parameters:
                file_path (str): Путь к файлу модели
                
            Returns:
                adaptive_model: Загруженная модель
            """
            try:
                # Загружаем данные из файла
                data = np.load(f"{file_path}.npz", allow_pickle=True)
                
                # Извлекаем основные параметры модели
                layer_sizes = data['layer_sizes'].tolist()
                activations = data['activations'].tolist()
                debug_settings = data['debug_settings'].tolist()
                
                # Создаем экземпляр модели
                model = cls(layer_sizes, activations, debug_settings)
                
                # Загружаем веса и смещения для каждого слоя
                for i in range(len(model.layers)):
                    model.layers[i]['weights'] = data[f'weights_{i}']
                    model.layers[i]['biases'] = data[f'biases_{i}']
                
                print(f"Model loaded from {file_path}")
                return model
            
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

# MARK: ===============



class nelin_text:
    """
    Comprehensive text processing toolkit for neural networks
    
    Features:
    - Tokenization (word and character level)
    - Text vectorization
    - Vocabulary management
    - Text encryption/decryption
    - Sequence padding
    - Text generation utilities
    """
    
    def __init__(self, text_corpus=None, mode='char'):
        """
        Initialize text processor
        
        Parameters:
            text_corpus (str/list): Text corpus for vocabulary building
            mode (str): Processing mode ('char' or 'word')
        """
        self.mode = mode
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.max_len = 0
        
        if text_corpus:
            self.build_vocab(text_corpus)
    
    def build_vocab(self, text_corpus):
        """
        Build vocabulary from text corpus
        
        Parameters:
            text_corpus (str/list): Text data for vocabulary
        """
        if isinstance(text_corpus, str):
            if self.mode == 'char':
                tokens = list(text_corpus)
            else:
                tokens = text_corpus.split()
        elif isinstance(text_corpus, list):
            tokens = [token for text in text_corpus for token in (list(text) if self.mode == 'char' else text.split())]
        else:
            raise ValueError("Unsupported corpus type. Use str or list.")
        
        # Create vocabulary
        unique_tokens = sorted(set(tokens))
        self.vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        # Add special tokens
        self.vocab['<PAD>'] = 0
        self.reverse_vocab[0] = '<PAD>'
        self.vocab['<UNK>'] = len(self.vocab)
        self.reverse_vocab[len(self.vocab)] = '<UNK>'
        
        self.vocab_size = len(self.vocab)
    
    def text_to_sequence(self, text, max_len=None):
        """
        Convert text to numerical sequence
        
        Parameters:
            text (str): Input text
            max_len (int): Maximum sequence length (padding/truncation)
        
        Returns:
            list: Numerical sequence
        """
        if self.mode == 'char':
            tokens = list(text)
        else:
            tokens = text.split()
        
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Update max length
        self.max_len = max(self.max_len, len(sequence))
        
        # Apply padding/truncation
        if max_len is None:
            max_len = self.max_len
        
        return self.pad_sequence(sequence, max_len)
    
    def sequence_to_text(self, sequence):
        """
        Convert numerical sequence back to text
        
        Parameters:
            sequence (list): Numerical sequence
            
        Returns:
            str: Reconstructed text
        """
        return ''.join(self.reverse_vocab.get(idx, '<UNK>') for idx in sequence if idx != 0 and idx in self.reverse_vocab)
    
    def pad_sequence(self, sequence, max_len):
        """
        Pad or truncate sequence to specified length
        
        Parameters:
            sequence (list): Numerical sequence
            max_len (int): Target length
            
        Returns:
            list: Padded/truncated sequence
        """
        if len(sequence) > max_len:
            return sequence[:max_len]
        return sequence + [0] * (max_len - len(sequence))
    
    def caesar_cipher(self, text, shift=3, encrypt=True):
        """
        Caesar cipher encryption/decryption
        
        Parameters:
            text (str): Input text
            shift (int): Shift value
            encrypt (bool): True for encryption, False for decryption
            
        Returns:
            str: Encrypted/decrypted text
        """
        result = []
        shift = shift if encrypt else -shift
        
        for char in text:
            if char.isalpha():
                base = ord('a') if char.islower() else ord('A')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def xor_cipher(self, text, key):
        """
        XOR cipher encryption/decryption
        
        Parameters:
            text (str): Input text
            key (str): Encryption key
            
        Returns:
            str: Encrypted/decrypted text
        """
        # Convert to bytes for processing
        text_bytes = text.encode('utf-8')
        key_bytes = key.encode('utf-8')
        
        # Repeat key to match text length
        repeated_key = (key_bytes * ((len(text_bytes) // len(key_bytes)) + 1))[:len(text_bytes)]
        
        # Apply XOR operation
        result = bytes([a ^ b for a, b in zip(text_bytes, repeated_key)])
        
        return result.decode('utf-8', errors='replace')
    
    def tokenize(self, text):
        """
        Tokenize text according to current mode
        
        Parameters:
            text (str): Input text
            
        Returns:
            list: Tokens
        """
        return list(text) if self.mode == 'char' else text.split()
    
    def generate_ngrams(self, text, n=2):
        """
        Generate n-grams from text
        
        Parameters:
            text (str): Input text
            n (int): Gram size
            
        Returns:
            list: n-grams
        """
        tokens = self.tokenize(text)
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def get_vocab_size(self):
        """
        Get current vocabulary size
        
        Returns:
            int: Number of tokens in vocabulary
        """
        return self.vocab_size
    
    def save_vocab(self, file_path):
        """
        Save vocabulary to file
        
        Parameters:
            file_path (str): Path to save vocabulary
        """
        import json
        with open(file_path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'reverse_vocab': {int(k): v for k, v in self.reverse_vocab.items()},
                'mode': self.mode,
                'max_len': self.max_len
            }, f)
    
    def load_vocab(self, file_path):
        """
        Load vocabulary from file
        
        Parameters:
            file_path (str): Path to vocabulary file
        """
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.reverse_vocab = {int(k): v for k, v in data['reverse_vocab'].items()}
            self.mode = data['mode']
            self.max_len = data['max_len']
            self.vocab_size = len(self.vocab)
# MARK: ===============



