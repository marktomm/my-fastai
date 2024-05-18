#!/usr/bin/env python
# coding: utf-8

import fastai.basics as fb

fb.plt.rc('figure', dpi=90)

def plot_function(f, title=None, min=-2.1, max=2.1, color='r', ylim=None):
    x = fb.torch.linspace(min, max, 100)[:,None]
    if ylim: 
        fb.plt.ylim(ylim)
    fb.plt.plot(x, f(x), color)
    if title is not None:
        fb.plt.title(title)

def f(x): return 3*x**2 + 2*x + 1
    
plot_function(f, '$3x^2 + 2x + 1$')

def quad(a, b, c, x): return a*x**2 + b*x + c
def mk_quad(a, b, c): return fb.partial(quad, a, b, c)
    
f2 = mk_quad(3, 2, 1)
plot_function(f2)
f(x).shape
def noise(x, scale): 
    return fb.np.random.normal(scale=scale, size=x.shape)
def add_noise(x, mult, add):
    return x * (1 + noise(x, mult)) + noise(x, add)

fb.np.random.seed(42)
x = fb.torch.linspace(-2, 2, steps=20)[:,None]
y = add_noise(f(x), 0.15, 1.5)
x[:5], y[:5]
fb.plt.scatter(x, y)

from ipywidgets import interact

@interact(a=1.1, b=1.1, c=1.1)
def plot_quad(a, b, c):
    fb.plt.scatter(x, y)
    plot_function(mk_quad(a,b,c), ylim=(-3,13))

def mae(preds, acts): return (fb.torch.abs(preds-acts)).mean()

@interact(a=1.1, b=1.1, c=1.1)
def plot_quad(a, b, c):
    f = mk_quad(a, b, c)
    fb.plt.scatter(x, y)
    loss = mae(f(x), y)
    plot_function(mk_quad(a, b, c), f'MAE: {loss:0.4f}', ylim=(-3, 13))

def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)

quad_mae([1.1, 1.1, 1.1])
abc = fb.torch.tensor([1.1,1.1,1.1])
abc.requires_grad_()
loss = quad_mae(abc)
loss
loss.backward()
print(abc.grad)