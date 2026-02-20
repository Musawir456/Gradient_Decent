## Gradient Descent Implementation in Python

This repository contains a simple Python implementation of gradient descent for linear regression. It demonstrates how to iteratively update the slope and intercept to minimize the cost function (Mean Squared Error).

## Features

Implements gradient descent from scratch using NumPy.

Calculates and prints the slope (m), intercept (b), and cost at each iteration.

Easy to modify for your own dataset.

Ideal for beginners learning machine learning basics.

## Dataset Example

You can use your own data arrays for x (input) and y (output).

import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
How to Run

Clone the repository:

git clone https://github.com/<your-username>/<repo-name>.git

Navigate into the project directory:

cd <repo-name>

## Run the Python script:

python gradient_descent.py

You will see output like this:

m 1.2, b 0.4, cost 4.5 iteration 0
m 1.5, b 0.7, cost 3.2 iteration 1
...
## Gradient Descent Function
def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print(f"m {m_curr}, b {b_curr}, cost {cost} iteration {i}")
## Requirements

Python 3.x

NumPy

Install dependencies using:

pip install numpy
## Author

Abdul Musawir – BS IT Student | AI/ML Engineer
