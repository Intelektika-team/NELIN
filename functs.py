import numpy as np
from typing import List, Tuple

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator
    NUMBA_AVAILABLE = False


def generate_xy(
    size: List[int] = [100, 5],  # [n_samples, n_features]
    noise_level: float = 0.1,     # Уровень шума в данных
    random_seed: int = None,      # Для воспроизводимости
    problem_type: str = "classification",  # "classification" или "regression"
    n_classes: int = 3,           # Количество классов (для классификации)
) -> Tuple[np.ndarray, np.ndarray]:
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples, n_features = size
    
    # Генерация признаков на основе комбинации функций
    X = np.random.randn(n_samples, n_features) * 2
    
    # Создаем полезные сигналы (признаки, которые влияют на y)
    for i in range(n_features):
        X[:, i] += i * np.sin(X[:, (i + 1) % n_features] * 0.5)
    
    # Добавляем шум
    X += np.random.normal(0, noise_level, size=(n_samples, n_features))
    
    if problem_type == "classification":
        # Создаем метки классов на основе признаков
        y = np.sum(X * np.arange(1, n_features + 1), axis=1)
        y = np.digitize(y, bins=np.linspace(np.min(y), np.max(y), n_classes))
    elif problem_type == "regression":
        # Создаем непрерывную целевую переменную
        y = np.sum(X ** 2, axis=1, keepdims=True) + np.random.normal(0, 0.5, size=(n_samples, 1))
    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")
    
    return X, y

# Функции ошибок - отлично подходят для JIT
@jit(nopython=True)
def st_error(now, true):
    """
    Calculate the simple difference error between true and predicted values.
    
    Parameters:
    now (array-like): Current predicted values
    true (array-like): Ground truth values
    
    Returns:
    array: Element-wise difference (true - now)
    """
    return true - now

@jit(nopython=True)
def nd_error(now, true):
    """
    Calculate the mean squared error (MSE) between true and predicted values.
    
    Parameters:
    now (array-like): Current predicted values
    true (array-like): Ground truth values
    
    Returns:
    float: Mean squared error
    """
    return np.mean((true - now)**2)

@jit(nopython=True)
def th_error(now, true):
    """
    Calculate the mean cubed error between true and predicted values.
    
    Parameters:
    now (array-like): Current predicted values
    true (array-like): Ground truth values
    
    Returns:
    float: Mean cubed error
    """
    return np.mean((true - now)**3)


def arr(inp):
    return np.array(inp)


@jit(nopython=True)
def sigmoid(x):
    """
    Sigmoid activation function - compresses values to range (0, 1)
    
    Parameters:
    x (array-like): Input values
    
    Returns:
    array: Sigmoid-transformed values
    """
    return 1 / (1 + np.exp(-x))

@jit(nopython=True)
def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function - returns 0 for negative inputs, 
    linear for positive inputs
    
    Parameters:
    x (array-like): Input values
    
    Returns:
    array: ReLU-transformed values
    """
    return np.maximum(0, x)

@jit(nopython=True)
def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function - modified ReLU with small slope for negative values
    
    Parameters:
    x (array-like): Input values
    alpha (float): Slope coefficient for negative values (default: 0.01)
    
    Returns:
    array: Leaky ReLU-transformed values
    """
    return np.where(x > 0, x, alpha * x)

@jit(nopython=True)
def tanh(x):
    """
    Hyperbolic tangent activation function - compresses values to range (-1, 1)
    
    Parameters:
    x (array-like): Input values
    
    Returns:
    array: tanh-transformed values
    """
    return np.tanh(x)

@jit(nopython=True)
def softmax(x):
    """
    Softmax activation function - converts vector to probability distribution
    
    Parameters:
    x (array-like): Input vector
    
    Returns:
    array: Probability distribution (sum = 1)
    """
    # Для стабильности вычислений
    max_val = np.max(x)
    e_x = np.exp(x - max_val)
    sum_e_x = np.sum(e_x)
    return e_x / sum_e_x

@jit(nopython=True)
def softsign(x):
    """
    Softsign activation function - smooth alternative to tanh
    
    Parameters:
    x (array-like): Input values
    
    Returns:
    array: Softsign-transformed values
    """
    return x / (1 + np.abs(x))

@jit(nopython=True)
def swish(x):
    """
    Swish activation function - modern activation function (x * sigmoid(x))
    
    Parameters:
    x (array-like): Input values
    
    Returns:
    array: Swish-transformed values
    """
    return x * sigmoid(x)

