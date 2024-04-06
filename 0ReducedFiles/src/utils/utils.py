
import numpy as np

########################################
#### DISTRIBUTIONS ########################
########################################

def sample_mixture_gaussian(bias, means=[-1.5, 1.5], std=[1.0, 1.0],  n_samples = 1000):
    """
    bias: between 0 and 1, weight over first distribution
    polarisation: between 0 and 1, how much the distribution are accentuated

    standard deviation : w/ sqrt
    scale variance is max variance
    """
    n1=int(n_samples*(1-bias))
    n2=int(n_samples*bias)

    x1 = np.random.normal(means[0], std[0], n1)
    x2 = np.random.normal(means[1], std[1], n2)

    X = np.array(list(x1) + list(x2))
    np.random.shuffle(X)
    print("Dataset shape:", X.shape)

    return X


def pdf(data, mean: float, variance: float, num_agents: int):
  # A normal continuous random variable.
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = num_agents * np.exp(-(np.square(data - mean)/(2*variance)))
  #scale it by num agents
  return s1 * s2

####################################
#### SAVE ########################
####################################

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, 
                          np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
