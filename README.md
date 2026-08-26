# Linear-Regression-Scratch
A Numpy only implementation of linear regression, built to revisit core ML mathematics (gradients, optimization) and rebuild Python fluency through hands on implementation rather than just simply calling libraries.

A from, scratch implementation of linear regression using only NumPy no scikit-learn, no autograd. Built to strengthen core ML fundamentals: gradients, optimization, and clean code structure, rather than to be a production-ready library.
scikit used only for comparison with custom class model and getting dataset.

## Why this project

Most ML workflows start with `model.fit(X, y)` and never look inside. This project goes the other direction — implementing the math and the training loop by hand to build (and verify) intuition for what's actually happening during training.
Also along with the iterative approach, also took time to understand the closed form approach of Linear Regression, along with it expanding the knowledge on iterative approach (i.e Gradient Descent) and Analytical approach (i.e Closed form).

## Structure

The code is deliberately split by responsibility, rather than one monolithic script:

- `model.py` — the `LinearRegression` class. Owns only the model's state (`w`, `b`) and knows how to `predict`. Has no knowledge of loss functions or optimizers.
- `loss_fn.py` — computes the loss value (MSE) and its gradient with respect to `w` and `b`. Kept separate from the model since the gradient formula is inherently tied to the choice of loss function, not to the model itself.
- `optimizer.py` — implements the parameter update rule (plain gradient descent). Takes gradients in, returns updated parameters, with no knowledge of how those gradients were derived.
- `train.py` — the training loop. The only file that ties model, loss, and optimizer together each epoch.
- `main.py` — loads data, initializes everything, runs training, and evaluates results.

This separation means, for example, swapping MSE for a different loss, or plain gradient descent for a different optimizer, doesn't require touching the model class at all.