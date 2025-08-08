import numpy as np
from nelin.functs import *
import random as r
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





version = "0.3.5"
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
    
    @property
    def device(self):
        """Current storage device ('cpu' or 'gpu')"""
        return self._device
    
    @property
    def shape(self):
        return self._array.shape
    
    @property
    def dtype(self):
        return self._array.dtype
    
    @property
    def nbytes(self):
        return self._array.nbytes
    
    def to(self, device):
        """Convert to specified device"""
        if device == self._device:
            return self
        return nlarr(self._array, device=device)
    
    def cpu(self):
        """Convert to CPU numpy array"""
        return self.to('cpu')
    
    def gpu(self):
        """Convert to GPU cupy array (if available)"""
        return self.to('gpu')
    
    def numpy(self):
        """Return as numpy array (always CPU)"""
        if self._device == 'gpu' and CUPY_AVAILABLE:
            return cp.asnumpy(self._array)
        return self._array
    
    def cupy(self):
        """Return as cupy array (if on GPU)"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        if self._device == 'cpu':
            return cp.asarray(self._array)
        return self._array
    
    def __array__(self):
        """Numpy array interface"""
        return self.numpy()
    
    def __cupy_array__(self):
        """CuPy array interface"""
        return self.cupy()
    
    def __repr__(self):
        return f"nlarr(device={self.device}, shape={self.shape}, dtype={self.dtype})\n" + repr(self._array)
    
    def __str__(self):
        return str(self._array)
    
    def __getitem__(self, key):
        return nlarr(self._array[key], device=self.device)
    
    def __setitem__(self, key, value):
        if isinstance(value, nlarr):
            self._array[key] = value.to(self.device)._array
        else:
            self._array[key] = value
    
    def __len__(self):
        return len(self._array)
    
    def __eq__(self, other):
        if isinstance(other, nlarr):
            return nlarr(self._array == other.to(self.device)._array, device=self.device)
        return nlarr(self._array == other, device=self.device)
    
    # Mathematical operations
    def __add__(self, other):
        return self._apply_op(other, '__add__')
    
    def __sub__(self, other):
        return self._apply_op(other, '__sub__')
    
    def __mul__(self, other):
        return self._apply_op(other, '__mul__')
    
    def __truediv__(self, other):
        return self._apply_op(other, '__truediv__')
    
    def __matmul__(self, other):
        return self._apply_op(other, '__matmul__')
    
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
    def sum(self, axis=None, keepdims=False):
        return nlarr(self._array.sum(axis=axis, keepdims=keepdims), device=self.device)
    
    def mean(self, axis=None, keepdims=False):
        return nlarr(self._array.mean(axis=axis, keepdims=keepdims), device=self.device)
    
    def max(self, axis=None, keepdims=False):
        return nlarr(self._array.max(axis=axis, keepdims=keepdims), device=self.device)
    
    def min(self, axis=None, keepdims=False):
        return nlarr(self._array.min(axis=axis, keepdims=keepdims), device=self.device)
    
    def transpose(self, axes=None):
        return nlarr(self._array.transpose(axes), device=self.device)
    
    def reshape(self, shape):
        return nlarr(self._array.reshape(shape), device=self.device)
    
    def copy(self):
        return nlarr(self._array.copy(), device=self.device)
    
    # Memory optimization
    def pin_memory(self):
        """Pin memory for faster GPU transfers (if available)"""
        if self.device == 'cpu' and CUPY_AVAILABLE:
            return cp.cuda.alloc_pinned_memory(self.nbytes)
        return self
    
    # Custom serialization
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
    def from_bytes(cls, header, data):
        """Deserialize from bytes"""
        shape = header['shape']
        dtype = np.dtype(header['dtype'])
        device = header['device']
        
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        return cls(array, device=device)
    
    # Factory methods
    @staticmethod
    def zeros(shape, dtype=np.float32, device='auto'):
        if device == 'auto':
            device = 'gpu' if CUPY_AVAILABLE else 'cpu'
            
        if device == 'gpu' and CUPY_AVAILABLE:
            return nlarr(cp.zeros(shape, dtype=dtype), device=device)
        return nlarr(np.zeros(shape, dtype=dtype), device=device)
    
    @staticmethod
    def ones(shape, dtype=np.float32, device='auto'):
        if device == 'auto':
            device = 'gpu' if CUPY_AVAILABLE else 'cpu'
            
        if device == 'gpu' and CUPY_AVAILABLE:
            return nlarr(cp.ones(shape, dtype=dtype), device=device)
        return nlarr(np.ones(shape, dtype=dtype), device=device)
    
    @staticmethod
    def empty(shape, dtype=np.float32, device='auto'):
        if device == 'auto':
            device = 'gpu' if CUPY_AVAILABLE else 'cpu'
            
        if device == 'gpu' and CUPY_AVAILABLE:
            return nlarr(cp.empty(shape, dtype=dtype), device=device)
        return nlarr(np.empty(shape, dtype=dtype), device=device)
    
    @staticmethod
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



