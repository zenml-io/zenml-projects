"""Simple calculator module for QualityFlow demonstration."""

from typing import Union


class Calculator:
    """A simple calculator with basic arithmetic operations."""

    def __init__(self):
        """Initialize calculator with empty history."""
        self.history = []

    def add(
        self, a: Union[int, float], b: Union[int, float]
    ) -> Union[int, float]:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(
        self, a: Union[int, float], b: Union[int, float]
    ) -> Union[int, float]:
        """Subtract second number from first."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(
        self, a: Union[int, float], b: Union[int, float]
    ) -> Union[int, float]:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(
        self, a: Union[int, float], b: Union[int, float]
    ) -> Union[int, float]:
        """Divide first number by second."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def power(
        self, base: Union[int, float], exponent: Union[int, float]
    ) -> Union[int, float]:
        """Raise base to the power of exponent."""
        result = base**exponent
        self.history.append(f"{base} ** {exponent} = {result}")
        return result

    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()

    def get_history(self) -> list[str]:
        """Get calculation history."""
        return self.history.copy()


def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
