from math import copysign


def sign(input_value: float) -> float:
    return copysign(1, input_value)