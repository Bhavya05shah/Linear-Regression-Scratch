import numpy as np

def update_wts(w, b, grad_w, grad_b, lr):
    w = w - lr*(grad_w)
    b = b - lr*(grad_b)
    return w,b 