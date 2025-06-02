import numpy as np
import math

def log_safe(x): return np.where(x <= 0, 0.0, np.log(x))

def sqrt_safe(x): return np.sqrt(np.abs(x))

def exp_safe(x): return np.exp(np.clip(x, -700, 700))

def div_safe(a, b): return np.where(np.abs(b) < 1e-12, 1.0, a / b)

def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return (
        div_safe(
            exp_safe(
                div_safe(
                    sqrt_safe(x[1]),
                    np.sin(x[2])
                ) / ((x[0] * -0.466) * exp_safe(x[2]))
            ),
            exp_safe(
                (div_safe(x[2], 4.004) * (6.319 + 7.442)) * ((-7.637 + x[2]) + log_safe(x[1]))
            )
        ) * np.cos(
            (
                sqrt_safe(x[0] * x[2]) * (div_safe(x[1], 9.200) * (x[0] * 4.569))
            ) + (
                div_safe(div_safe(x[0], x[0]), -x[1]) * ((x[1] * x[2]) + (x[2] + 3.061))
            )
        )
    )


def f3(x: np.ndarray) -> np.ndarray:
    return (
        div_safe(x[2], 9.142) +
        (
            sqrt_safe(-9.933 - 5.874) +
            (x[0] * x[0]) +
            ((-x[1] * (x[1] * x[1])) + ((x[0] * x[0]) - (3.600 * x[2])))
        )
    )


def f4(x: np.ndarray) -> np.ndarray:
    return (
        div_safe(
            div_safe(x[0], -9.239),
            div_safe((x[0] * 8.815), div_safe(x[0], x[0]))
        ) +
        div_safe(x[0], -1.188) / np.abs(9.368) +
        ((np.cos(x[1]) * np.abs(7.003)) + (6.538 * 0.504))
    )


def f5(x: np.ndarray) -> np.ndarray:
    return (
        (
            exp_safe(x[1]) +
            (8.736 + x[1]) +
            exp_safe(x[0]) +
            (
                div_safe(exp_safe(x[0]), (x[1] - 5.465)) * (x[1] * x[1])
            )
        ) /
        (
            (
                exp_safe((x[1] * x[1]) - np.abs(x[0])) *
                np.abs(div_safe(exp_safe(6.353), (-5.603 - x[1])))
            ) +
            (
                (x[0] * x[1]) +
                (4.408 * x[0]) +
                (div_safe(x[1], -2.113) * (2.529 * x[1])) +
                exp_safe(div_safe(-3.779, -0.145))
            )
        )
    )


def f6(x: np.ndarray) -> np.ndarray:
    return (
        (
            (x[0] * -0.695) + (1.647 * x[1])
        )
        - div_safe(
            1.647 * x[1],
            (-4.338 * 9.012) - (-4.442)
        )
        - div_safe(
            div_safe(
                x[0] / -5.559,
                (-4.338 * 9.012) - (-4.442)
            ),
            (
                (
                    ((x[1] - x[1]) * (x[1] * x[1])) *
                    ((-x[0]) + (x[0] * 7.529))
                ) -
                (
                    ((x[1] + x[1]) + (9.243 * 3.343)) -
                    ((x[1] - x[0]) - (x[1] * x[1]))
                )
            )
        )
    )


def f7(x: np.ndarray) -> np.ndarray:
    return (
        np.abs(log_safe(np.abs((x[1] - x[0]) * np.abs(x[0])))) *
        (
            ((x[1] * x[0]) + (np.abs(x[0]) + (x[0] * x[0]))) +
            (x[1] * x[0]) +
            exp_safe((3.473 - 2.375) + (x[1] * x[0]))
        )
    )


def f8(x: np.ndarray) -> np.ndarray:
    return (
        (
            (
                (
                    (9.193 * x[3]) - (x[2] + x[4])
                ) - (
                    (x[3] + -9.069) + (x[1] - x[3])
                )
            ) + (
                log_safe(np.abs(x[5] - x[4])) - (x[5] * x[5])
            )
        ) + (
            ((x[5] * x[5]) * (x[5] * x[5])) * (4.955 * x[5])
        )
    ) - np.abs(
        (
            (x[4] * 9.044) + ((-3.330 - x[1]) - (x[4] * -8.919))
        ) * (x[4] * x[4])
    )



def prepare_data(file_path):
    """
    Prepares the data for Symbolic Regression with Genetic Programming
   
    Args:
        file_path: Path to the .npz file containing the data
       
    Returns:
        X: Array of input features
        y: Array of target outputs
        config: Configuration for GP based on the data
    """
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    X = data['x']
    y = data['y']
   
    if X.ndim > 1 and X.shape[1] > X.shape[0] and y.shape[0] == X.shape[1]:
        print("Detected format with features on rows and samples on columns, transposition...")
        X = X.T  
   
    if y.ndim > 1:
        if y.shape[0] == 1 or y.shape[1] == 1:  
            y = y.flatten()
            print(f"y transformed into form {y.shape}")
   
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        print(f"X rendered 2D with shape{X.shape}")
   
    print(f"Final form: X shape {X.shape}, y shape {y.shape}")
   
    if X.shape[0] != len(y):
        raise ValueError(f"Inconsistent number of samples: X ha {X.shape[0]} campioni, y ne ha {len(y)}")
   
    n_features = X.shape[1]
    print(f"Input {n_features}-dimensional with {X.shape[0]} samples")
   
    variables = [f'x[{i}]' for i in range(n_features)]
   
    const_range = max(np.max(np.abs(X)), np.max(np.abs(y)))
   
    config = {
        'variables': variables,
        'n_features': n_features,
        'const_range': const_range,
        'y_stats': {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y))
        },
        'dataset_size': len(y)
    }
   
    return X, y, config

def calculate_mse(f_pred, X, y_true):
    """
    Calcola l'errore quadratico medio (MSE) tra le predizioni di una funzione e i valori reali.
   
    Args:
        f_pred: Funzione che produce le predizioni
        X: Array numpy con i dati di input (caratteristiche)
        y_true: Array numpy con i valori target reali
       
    Returns:
        float: Il valore MSE calcolato
    """
    try:
        y_pred = np.array([f_pred(x) for x in X])
       
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('inf')
           
        mse = np.mean((y_pred - y_true) ** 2)
        return mse
    except Exception as e:
        print(f"Errore nel calcolo delle predizioni: {e}")
        return float('inf')
 
def evaluate_corresponding_functions():
    """
    Valuta ogni funzione sul suo problema corrispondente:
    f0 su problem_0, f1 su problem_1, ecc.
    """
    functions = {
        1: f1,
        2: f2,
        3: f3,
        4: f4,
        5: f5,
        6: f6,
        7: f7,
        8: f8
    }
   
    results = []
   
    for i in range(8):
        problem_path = f"C:\\Users\\Sergio\\Desktop\\New CI Project\\CI2024_project-work\\data\\problem_{i+1}.npz"
        function = functions[i+1]
       
        try:
            X, y, config = prepare_data(problem_path)
           
            mse = calculate_mse(function, X, y)
           
            print(f"MSE della funzione f{i+1} sul problema {i+1}: {mse}")
            results.append((i+1, mse))
        except Exception as e:
            print(f"Errore nella valutazione della funzione f{i+1} sul problema {i+1}: {e}")
            results.append((i+1, float('inf')))
   
    print("\nRiepilogo dei risultati:")
    print("-" * 50)
    print("| Problema | Funzione | MSE |")
    print("|----------|----------|-----------------|")
    for prob_idx, mse in results:
        if np.isinf(mse):
            print(f"| problem_{prob_idx} | f{prob_idx} | inf |")
        else:
            print(f"| problem_{prob_idx} | f{prob_idx} | {mse:.6e} |")
    print("-" * 50)

if __name__ == "__main__":
    print("Inizio della valutazione delle funzioni...")
    print("-" * 50)
    evaluate_corresponding_functions()

