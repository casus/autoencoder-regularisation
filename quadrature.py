import numpy as np
import itertools

# Generate quadrature points and weights with Legendre-Gauss('LL') or Chebyshev-Gauss('CC')
def getQuad(quad='LL', npoints=3, ndim=2):
  if (quad=='LL'):
    q, w = np.polynomial.legendre.leggauss(npoints)
  elif (quad=='CC'):
    q, wtemp = np.polynomial.chebyshev.chebgauss(npoints)
    w = wtemp * np.sqrt(1 - (q[:])**2) ## Since weight function has f(x) = 1/sqrt{1 - x^2}
  else:
    print("No points and weights gerenated since quadrature not specified!")
    return np.array([]), np.array([])

  points = np.array(list(itertools.product(q, repeat=ndim)))
  weights = np.prod(np.array(list(itertools.product(w, repeat=ndim))), axis=1)
  return points, weights

# Transformation
def transform(points, weights, limits):
  if (points.shape[0] > 0):
    transfromPoints = np.zeros(points.shape)
    transfromWeights = weights
    for counter, limit in enumerate(limits):
      transfromPoints[:,counter] = (limit[1] - limit[0])/2 * points[:,counter] + (limit[0] + limit[1])/2
      transfromWeights = transfromWeights * (limit[1] - limit[0])/2
    return transfromPoints, transfromWeights
  else:
    print("No transformed points and weights gerenated!")
    return np.array([]), np.array([])

# Integrate
def solveQuad(fn, limits, quad, npoints):
  ndim = len(limits)
  if (ndim > 0):
    points, weights = getQuad(quad, npoints, ndim)
    q, w = transform(points, weights, limits)
    if (q.shape[0] > 0):
      return np.sum(fn(q) * w[:])
    else:
      print("No solution gerenated!")
  else:
    print("No solution gerenated since limits and dimensions are less than 1!")
