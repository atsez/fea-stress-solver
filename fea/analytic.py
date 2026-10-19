"""Reference solutions used for verification and validation.

Kirsch (1898). Infinite plate, circular hole of radius a, remote uniaxial stress s0 along y.
Polar angle theta is measured from the x axis, which is perpendicular to the load, so the peak
hoop stress 3 * s0 sits at theta = 0, the point (a, 0).

Finite-width plate with a central circular hole, net-section stress concentration factor
Ktn = peak stress / (P / ((W - d) t)), d = 2a:
  * Howland (1930) numerical solution as fitted in Pilkey and Pilkey, Peterson's Stress
    Concentration Factors, 3rd ed. (Wiley, 2008), chapter 4:
        Ktn = 2 + 0.284 (1 - d/W) - 0.600 (1 - d/W)^2 + 1.32 (1 - d/W)^3
  * Heywood (1952), Designing by Photoelasticity:
        Ktn = 2 + (1 - d/W)^3
  * Polynomial quoted in handbooks (Roark form):
        Ktn = 3.000 - 3.140 (d/W) + 3.667 (d/W)^2 - 1.527 (d/W)^3
Gross-section factor Ktg = Ktn / (1 - d/W).

Inglis (1913). Infinite plate, elliptical hole with semi-axis a perpendicular to the load and b
along it: Kt = 1 + 2 a / b.
"""
import numpy as np


def kirsch(x, y, a, s0=1.0):
    """Cartesian stress (n, 3) of the Kirsch solution, load s0 along y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = x ** 2 + y ** 2
    th = np.arctan2(y, x)
    q2, q4 = a ** 2 / r2, a ** 4 / r2 ** 2
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    srr = s0 / 2 * (1 - q2) - s0 / 2 * (1 - 4 * q2 + 3 * q4) * c2
    stt = s0 / 2 * (1 + q2) + s0 / 2 * (1 + 3 * q4) * c2
    srt = s0 / 2 * (1 + 2 * q2 - 3 * q4) * s2
    c, s = np.cos(th), np.sin(th)
    sxx = srr * c ** 2 + stt * s ** 2 - 2 * srt * s * c
    syy = srr * s ** 2 + stt * c ** 2 + 2 * srt * s * c
    sxy = (srr - stt) * s * c + srt * (c ** 2 - s ** 2)
    return np.stack([sxx, syy, sxy], axis=-1)


def ktn_howland(dW):
    q = 1 - np.asarray(dW, float)
    return 2 + 0.284 * q - 0.600 * q ** 2 + 1.32 * q ** 3


def ktn_heywood(dW):
    return 2 + (1 - np.asarray(dW, float)) ** 3


def ktn_poly(dW):
    x = np.asarray(dW, float)
    return 3.000 - 3.140 * x + 3.667 * x ** 2 - 1.527 * x ** 3


def kt_inglis(a, b):
    return 1 + 2 * a / b


def bending_exact(x, y, k, E, nu):
    """Pure bending, sxx = k y. Returns (ux, uy) for plane stress with ux(0, y) = 0, uy(0, 0) = 0."""
    return k * x * y / E, -k / (2 * E) * (x ** 2 + nu * y ** 2)
