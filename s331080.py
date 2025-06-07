import numpy as np

def log_safe(x): return np.where(x <= 0, 0.0, np.log(x))

def sqrt_safe(x): return np.sqrt(np.abs(x))

def exp_safe(x): return np.exp(np.clip(x, -700, 700))

def div_safe(a, b): return np.where(np.abs(b) < 1e-12, 1.0, a / b)

# Problem 1
# MSE: 7.125940794232773e-34
def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

# Problem 2
# MSE: 10000000000.0
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

# Problem 3
# MSE: 0.000133798610577
def f3(x: np.ndarray) -> np.ndarray:
    return (
        (
            np.sin(
                (
                    (
                        (x[2] - x[2]) / (
                            (
                                np.log(np.abs(x[1])) - (x[1] * x[2])
                            ) / (
                                (-9.103 - x[2]) * (
                                    (x[2] / x[1]) + np.abs(x[0])
                                )
                            )
                        )
                    ) + np.cos(6.457)
                )
            )
        ) + (
            (x[0] * x[0]) + (
                ((x[2] * -2.504) + ((x[0] * x[2]) * (x[0] / x[2]))) +
                (((x[1] * x[1]) * (-x[1])) - (-3.167 + x[2]))
            )
        )
    )

# Problem 4
# MSE: 1.495061516474593e-05
def f4(x: np.ndarray) -> np.ndarray:
    return (
        div_safe(
            div_safe(x[0], -9.239),
            div_safe((x[0] * 8.815), div_safe(x[0], x[0]))
        ) +
        div_safe(x[0], -1.188) / np.abs(9.368) +
        ((np.cos(x[1]) * np.abs(7.003)) + (6.538 * 0.504))
    )

# Problem 5
# MSE: 2.651652350332708e-20
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

# Problem 6
# MSE: 8.251367834469018e-11
def f6(x: np.ndarray) -> np.ndarray:
    return (
        div_safe(
            (
                ((x[0] - 0.198) + ((9.979 * -4.804) * (x[1] + x[0]))) -
                (x[0] / -1.434) + (-9.896 * x[1])
            ),
            (
                div_safe(
                    ((x[0] - x[1]) * (x[0] / -1.434)),
                    (
                        ((-1.538 + x[1]) * exp_safe(x[0])) - np.sin(x[0] * x[1]) +
                        exp_safe(9.682)
                    )
                ) + (
                    ((x[0] - x[0]) + log_safe((x[0] - x[1]) + (x[1] / x[0]))) -
                    (
                        (np.sin(x[1] + x[0]) + exp_safe(9.682)) -
                        (
                            ((8.148 - 7.531) + (-9.831 * -7.634)) -
                            ((x[0] + x[0]) - (x[0] + 1.343))
                        )
                    )
                )
            )
        ) + ((x[0] / -1.434) + (x[1] * 1.691))
    )

# Problem 7
# MSE: 50.19040470784147
def f7(x: np.ndarray) -> np.ndarray:
    return (
        np.abs(log_safe(np.abs((x[1] - x[0]) * np.abs(x[0])))) *
        (
            ((x[1] * x[0]) + (np.abs(x[0]) + (x[0] * x[0]))) +
            (x[1] * x[0]) +
            exp_safe((3.473 - 2.375) + (x[1] * x[0]))
        )
    )

# Problem 8
# MSE: 27568.656269303516
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