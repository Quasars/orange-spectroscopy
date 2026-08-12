Unwrap signal
=============

This widget is a wrapper for the [NumPy signal unwrapping function](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html). It uses the default arguments of:

```
wrapped signal = unwrap(input signal, discont = pi, period = 2*pi)
```

This unwrapping is calculated for every row of the input data table.
