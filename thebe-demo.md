---
layout: default
title: ThebeLab Demo
---

# ThebeLab Demo: Interactive memo Examples

This page demonstrates executable code blocks using ThebeLab. Click "Activate Interactive Code" in the top right, then click the "Run" buttons to execute code.

## Test Basic Python

```python
# Test if basic Python works
import numpy as np
import matplotlib.pyplot as plt

print("✅ Basic Python working!")
print("NumPy version:", np.__version__)

# Simple computation
x = np.arange(10)
y = x ** 2
print("Simple array computation:")
print("x =", x)
print("y = x^2 =", y)
```
{: data-executable="true"}

## Basic memo Example

```python
# Try memo - this might fail if Binder can't build JAX
try:
    import memo
    import numpy as np

    # Simple example with memo
    X = np.arange(5)

    @memo.memo
    def simple_model[a: X, b: X]():
        return a + b

    # Calculate the result
    result = simple_model()
    print("✅ memo working!")
    print("Result shape:", result.shape)
    print("Result values:")
    print(result)
except ImportError as e:
    print("❌ memo not available:", e)
    print("This is expected if JAX couldn't be installed in the Binder environment")
```
{: data-executable="true"}

## JAX Integration

```python
import jax
import jax.numpy as jnp
import memo

# JAX-based computation
@memo.memo  
def jax_example[n: jnp.arange(3, 8)]():
    x = jnp.arange(n)
    return jnp.sum(x ** 2)

result = jax_example()
print("JAX computation results:")
print(result)
```
{: data-executable="true"}

## Interactive Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
```
{: data-executable="true"}

## Instructions

1. Click the **"Activate Interactive Code"** button in the top right corner
2. Wait for the kernel to start (this may take 1-2 minutes on first load)
3. Click the **"▶ Run"** button above any code block to execute it
4. The output will appear below the code block

The kernel runs on Binder and has access to all packages specified in `requirements.txt`. 