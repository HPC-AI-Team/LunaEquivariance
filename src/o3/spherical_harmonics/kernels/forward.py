"""Cartesian spherical-harmonic CUDA, degrees 0 through 12.

CARTESIAN_POLYNOMIAL is the single source for the forward arithmetic.
Forward parses it once; backward differentiates the same ordered operations.
The scalar emitter, CUDA loop and binding are reused by all derivative orders."""

from __future__ import annotations

import ast
from functools import lru_cache
import math
from math import prod

import torch

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension
from ....utils.precision import float_literal
from .scalar import ScalarGraph, cuda_assignments, lower_expression
from .utils import HOST_BINDING


CARTESIAN_POLYNOMIAL = r"""
// Degree 0
const Element sh_0_0 = Element(1.0);

// Degree 1
const Element sh_1_0 = Element(std::sqrt(3.0)) * x;
const Element sh_1_1 = Element(std::sqrt(3.0)) * y;
const Element sh_1_2 = Element(std::sqrt(3.0)) * z;

// Degree 2
const Element sh_2_0 = Element(std::sqrt(15.0)) * x * z;
const Element sh_2_1 = Element(std::sqrt(15.0)) * x * y;
const Element y2 = Element(std::pow(y, 2));
const Element x2z2 = Element(std::pow(x, 2)) + Element(std::pow(z, 2));
const Element sh_2_2 = Element(std::sqrt(5.0)) * (y2 - Element(1.0 / 2.0) * x2z2);
const Element sh_2_3 = Element(std::sqrt(15.0)) * y * z;
const Element sh_2_4 = Element(1.0 / 2.0 * std::sqrt(15.0)) * (Element(std::pow(z, 2)) - Element(std::pow(x, 2)));

// Degree 3
const Element sh_3_0 = Element(1.0 / 6.0 * std::sqrt(42.0)) * (sh_2_0 * z + sh_2_4 * x);
const Element sh_3_1 = Element(std::sqrt(7.0)) * sh_2_0 * y;
const Element sh_3_2 = Element(1.0 / 8.0 * std::sqrt(168.0)) * (Element(4.0) * y2 - x2z2) * x;
const Element sh_3_3 = Element(1.0 / 2.0 * std::sqrt(7.0)) * y * (Element(2.0) * y2 - Element(3.0) * x2z2);
const Element sh_3_4 = Element(1.0 / 8.0 * std::sqrt(168.0)) * z * (Element(4.0) * y2 - x2z2);
const Element sh_3_5 = Element(std::sqrt(7.0)) * sh_2_4 * y;
const Element sh_3_6 = Element(1.0 / 6.0 * std::sqrt(42.0)) * (sh_2_4 * z - sh_2_0 * x);

// Degree 4
const Element sh_4_0 = Element(3.0 / 4.0 * std::sqrt(2.0)) * (sh_3_0 * z + sh_3_6 * x);
const Element sh_4_1 = Element(3.0 / 4.0) * sh_3_0 * y
    + Element(3.0 / 8.0 * std::sqrt(6.0)) * sh_3_1 * z
    + Element(3.0 / 8.0 * std::sqrt(6.0)) * sh_3_5 * x;
const Element sh_4_2 = Element(-3.0 / 56.0 * std::sqrt(14.0)) * sh_3_0 * z
    + Element(3.0 / 14.0 * std::sqrt(21.0)) * sh_3_1 * y
    + Element(3.0 / 56.0 * std::sqrt(210.0)) * sh_3_2 * z
    + Element(3.0 / 56.0 * std::sqrt(210.0)) * sh_3_4 * x
    + Element(3.0 / 56.0 * std::sqrt(14.0)) * sh_3_6 * x;
const Element sh_4_3 = Element(-3.0 / 56.0 * std::sqrt(42.0)) * sh_3_1 * z
    + Element(3.0 / 28.0 * std::sqrt(105.0)) * sh_3_2 * y
    + Element(3.0 / 28.0 * std::sqrt(70.0)) * sh_3_3 * x
    + Element(3.0 / 56.0 * std::sqrt(42.0)) * sh_3_5 * x;
const Element sh_4_4 = Element(-3.0 / 28.0 * std::sqrt(42.0)) * sh_3_2 * x
    + Element(3.0 / 7.0 * std::sqrt(7.0)) * sh_3_3 * y
    - Element(3.0 / 28.0 * std::sqrt(42.0)) * sh_3_4 * z;
const Element sh_4_5 = Element(-3.0 / 56.0 * std::sqrt(42.0)) * sh_3_1 * x
    + Element(3.0 / 28.0 * std::sqrt(70.0)) * sh_3_3 * z
    + Element(3.0 / 28.0 * std::sqrt(105.0)) * sh_3_4 * y
    - Element(3.0 / 56.0 * std::sqrt(42.0)) * sh_3_5 * z;
const Element sh_4_6 = Element(-3.0 / 56.0 * std::sqrt(14.0)) * sh_3_0 * x
    - Element(3.0 / 56.0 * std::sqrt(210.0)) * sh_3_2 * x
    + Element(3.0 / 56.0 * std::sqrt(210.0)) * sh_3_4 * z
    + Element(3.0 / 14.0 * std::sqrt(21.0)) * sh_3_5 * y
    - Element(3.0 / 56.0 * std::sqrt(14.0)) * sh_3_6 * z;
const Element sh_4_7 = Element(-3.0 / 8.0 * std::sqrt(6.0)) * sh_3_1 * x
    + Element(3.0 / 8.0 * std::sqrt(6.0)) * sh_3_5 * z
    + Element(3.0 / 4.0) * sh_3_6 * y;
const Element sh_4_8 = Element(3.0 / 4.0 * std::sqrt(2.0)) * (-sh_3_0 * x + sh_3_6 * z);

// Degree 5
const Element sh_5_0 = Element(1.0 / 10.0 * std::sqrt(110.0)) * (sh_4_0 * z + sh_4_8 * x);
const Element sh_5_1 = Element(1.0 / 5.0 * std::sqrt(11.0)) * sh_4_0 * y
    + Element(1.0 / 5.0 * std::sqrt(22.0)) * sh_4_1 * z
    + Element(1.0 / 5.0 * std::sqrt(22.0)) * sh_4_7 * x;
const Element sh_5_2 = Element(-1.0 / 30.0 * std::sqrt(22.0)) * sh_4_0 * z
    + Element(4.0 / 15.0 * std::sqrt(11.0)) * sh_4_1 * y
    + Element(1.0 / 15.0 * std::sqrt(154.0)) * sh_4_2 * z
    + Element(1.0 / 15.0 * std::sqrt(154.0)) * sh_4_6 * x
    + Element(1.0 / 30.0 * std::sqrt(22.0)) * sh_4_8 * x;
const Element sh_5_3 = Element(-1.0 / 30.0 * std::sqrt(66.0)) * sh_4_1 * z
    + Element(1.0 / 15.0 * std::sqrt(231.0)) * sh_4_2 * y
    + Element(1.0 / 30.0 * std::sqrt(462.0)) * sh_4_3 * z
    + Element(1.0 / 30.0 * std::sqrt(462.0)) * sh_4_5 * x
    + Element(1.0 / 30.0 * std::sqrt(66.0)) * sh_4_7 * x;
const Element sh_5_4 = Element(-1.0 / 15.0 * std::sqrt(33.0)) * sh_4_2 * z
    + Element(2.0 / 15.0 * std::sqrt(66.0)) * sh_4_3 * y
    + Element(1.0 / 15.0 * std::sqrt(165.0)) * sh_4_4 * x
    + Element(1.0 / 15.0 * std::sqrt(33.0)) * sh_4_6 * x;
const Element sh_5_5 = Element(-1.0 / 15.0 * std::sqrt(110.0)) * sh_4_3 * x
    + Element(1.0 / 3.0 * std::sqrt(11.0)) * sh_4_4 * y
    - Element(1.0 / 15.0 * std::sqrt(110.0)) * sh_4_5 * z;
const Element sh_5_6 = Element(-1.0 / 15.0 * std::sqrt(33.0)) * sh_4_2 * x
    + Element(1.0 / 15.0 * std::sqrt(165.0)) * sh_4_4 * z
    + Element(2.0 / 15.0 * std::sqrt(66.0)) * sh_4_5 * y
    - Element(1.0 / 15.0 * std::sqrt(33.0)) * sh_4_6 * z;
const Element sh_5_7 = Element(-1.0 / 30.0 * std::sqrt(66.0)) * sh_4_1 * x
    - Element(1.0 / 30.0 * std::sqrt(462.0)) * sh_4_3 * x
    + Element(1.0 / 30.0 * std::sqrt(462.0)) * sh_4_5 * z
    + Element(1.0 / 15.0 * std::sqrt(231.0)) * sh_4_6 * y
    - Element(1.0 / 30.0 * std::sqrt(66.0)) * sh_4_7 * z;
const Element sh_5_8 = Element(-1.0 / 30.0 * std::sqrt(22.0)) * sh_4_0 * x
    - Element(1.0 / 15.0 * std::sqrt(154.0)) * sh_4_2 * x
    + Element(1.0 / 15.0 * std::sqrt(154.0)) * sh_4_6 * z
    + Element(4.0 / 15.0 * std::sqrt(11.0)) * sh_4_7 * y
    - Element(1.0 / 30.0 * std::sqrt(22.0)) * sh_4_8 * z;
const Element sh_5_9 = Element(-1.0 / 5.0 * std::sqrt(22.0)) * sh_4_1 * x
    + Element(1.0 / 5.0 * std::sqrt(22.0)) * sh_4_7 * z
    + Element(1.0 / 5.0 * std::sqrt(11.0)) * sh_4_8 * y;
const Element sh_5_10 = Element(1.0 / 10.0 * std::sqrt(110.0)) * (-sh_4_0 * x + sh_4_8 * z);

// Degree 6
const Element sh_6_0 = Element(1.0 / 6.0 * std::sqrt(39.0)) * (sh_5_0 * z + sh_5_10 * x);
const Element sh_6_1 = Element(1.0 / 6.0 * std::sqrt(13.0)) * sh_5_0 * y
    + Element(1.0 / 12.0 * std::sqrt(130.0)) * sh_5_1 * z
    + Element(1.0 / 12.0 * std::sqrt(130.0)) * sh_5_9 * x;
const Element sh_6_2 = Element(-1.0 / 132.0 * std::sqrt(286.0)) * sh_5_0 * z
    + Element(1.0 / 33.0 * std::sqrt(715.0)) * sh_5_1 * y
    + Element(1.0 / 132.0 * std::sqrt(286.0)) * sh_5_10 * x
    + Element(1.0 / 44.0 * std::sqrt(1430.0)) * sh_5_2 * z
    + Element(1.0 / 44.0 * std::sqrt(1430.0)) * sh_5_8 * x;
const Element sh_6_3 = Element(-1.0 / 132.0 * std::sqrt(858.0)) * sh_5_1 * z
    + Element(1.0 / 22.0 * std::sqrt(429.0)) * sh_5_2 * y
    + Element(1.0 / 22.0 * std::sqrt(286.0)) * sh_5_3 * z
    + Element(1.0 / 22.0 * std::sqrt(286.0)) * sh_5_7 * x
    + Element(1.0 / 132.0 * std::sqrt(858.0)) * sh_5_9 * x;
const Element sh_6_4 = Element(-1.0 / 66.0 * std::sqrt(429.0)) * sh_5_2 * z
    + Element(2.0 / 33.0 * std::sqrt(286.0)) * sh_5_3 * y
    + Element(1.0 / 66.0 * std::sqrt(2002.0)) * sh_5_4 * z
    + Element(1.0 / 66.0 * std::sqrt(2002.0)) * sh_5_6 * x
    + Element(1.0 / 66.0 * std::sqrt(429.0)) * sh_5_8 * x;
const Element sh_6_5 = Element(-1.0 / 66.0 * std::sqrt(715.0)) * sh_5_3 * z
    + Element(1.0 / 66.0 * std::sqrt(5005.0)) * sh_5_4 * y
    + Element(1.0 / 66.0 * std::sqrt(3003.0)) * sh_5_5 * x
    + Element(1.0 / 66.0 * std::sqrt(715.0)) * sh_5_7 * x;
const Element sh_6_6 = Element(-1.0 / 66.0 * std::sqrt(2145.0)) * sh_5_4 * x
    + Element(1.0 / 11.0 * std::sqrt(143.0)) * sh_5_5 * y
    - Element(1.0 / 66.0 * std::sqrt(2145.0)) * sh_5_6 * z;
const Element sh_6_7 = Element(-1.0 / 66.0 * std::sqrt(715.0)) * sh_5_3 * x
    + Element(1.0 / 66.0 * std::sqrt(3003.0)) * sh_5_5 * z
    + Element(1.0 / 66.0 * std::sqrt(5005.0)) * sh_5_6 * y
    - Element(1.0 / 66.0 * std::sqrt(715.0)) * sh_5_7 * z;
const Element sh_6_8 = Element(-1.0 / 66.0 * std::sqrt(429.0)) * sh_5_2 * x
    - Element(1.0 / 66.0 * std::sqrt(2002.0)) * sh_5_4 * x
    + Element(1.0 / 66.0 * std::sqrt(2002.0)) * sh_5_6 * z
    + Element(2.0 / 33.0 * std::sqrt(286.0)) * sh_5_7 * y
    - Element(1.0 / 66.0 * std::sqrt(429.0)) * sh_5_8 * z;
const Element sh_6_9 = Element(-1.0 / 132.0 * std::sqrt(858.0)) * sh_5_1 * x
    - Element(1.0 / 22.0 * std::sqrt(286.0)) * sh_5_3 * x
    + Element(1.0 / 22.0 * std::sqrt(286.0)) * sh_5_7 * z
    + Element(1.0 / 22.0 * std::sqrt(429.0)) * sh_5_8 * y
    - Element(1.0 / 132.0 * std::sqrt(858.0)) * sh_5_9 * z;
const Element sh_6_10 = Element(-1.0 / 132.0 * std::sqrt(286.0)) * sh_5_0 * x
    - Element(1.0 / 132.0 * std::sqrt(286.0)) * sh_5_10 * z
    - Element(1.0 / 44.0 * std::sqrt(1430.0)) * sh_5_2 * x
    + Element(1.0 / 44.0 * std::sqrt(1430.0)) * sh_5_8 * z
    + Element(1.0 / 33.0 * std::sqrt(715.0)) * sh_5_9 * y;
const Element sh_6_11 = Element(-1.0 / 12.0 * std::sqrt(130.0)) * sh_5_1 * x
    + Element(1.0 / 6.0 * std::sqrt(13.0)) * sh_5_10 * y
    + Element(1.0 / 12.0 * std::sqrt(130.0)) * sh_5_9 * z;
const Element sh_6_12 = Element(1.0 / 6.0 * std::sqrt(39.0)) * (-sh_5_0 * x + sh_5_10 * z);

// Degree 7
const Element sh_7_0 = Element(1.0 / 14.0 * std::sqrt(210.0)) * (sh_6_0 * z + sh_6_12 * x);
const Element sh_7_1 = Element(1.0 / 7.0 * std::sqrt(15.0)) * sh_6_0 * y
    + Element(3.0 / 7.0 * std::sqrt(5.0)) * sh_6_1 * z
    + Element(3.0 / 7.0 * std::sqrt(5.0)) * sh_6_11 * x;
const Element sh_7_2 = Element(-1.0 / 182.0 * std::sqrt(390.0)) * sh_6_0 * z
    + Element(6.0 / 91.0 * std::sqrt(130.0)) * sh_6_1 * y
    + Element(3.0 / 91.0 * std::sqrt(715.0)) * sh_6_10 * x
    + Element(1.0 / 182.0 * std::sqrt(390.0)) * sh_6_12 * x
    + Element(3.0 / 91.0 * std::sqrt(715.0)) * sh_6_2 * z;
const Element sh_7_3 = Element(-3.0 / 182.0 * std::sqrt(130.0)) * sh_6_1 * z
    + Element(3.0 / 182.0 * std::sqrt(130.0)) * sh_6_11 * x
    + Element(3.0 / 91.0 * std::sqrt(715.0)) * sh_6_2 * y
    + Element(5.0 / 182.0 * std::sqrt(858.0)) * sh_6_3 * z
    + Element(5.0 / 182.0 * std::sqrt(858.0)) * sh_6_9 * x;
const Element sh_7_4 = Element(3.0 / 91.0 * std::sqrt(65.0)) * sh_6_10 * x
    - Element(3.0 / 91.0 * std::sqrt(65.0)) * sh_6_2 * z
    + Element(10.0 / 91.0 * std::sqrt(78.0)) * sh_6_3 * y
    + Element(15.0 / 182.0 * std::sqrt(78.0)) * sh_6_4 * z
    + Element(15.0 / 182.0 * std::sqrt(78.0)) * sh_6_8 * x;
const Element sh_7_5 = Element(-5.0 / 91.0 * std::sqrt(39.0)) * sh_6_3 * z
    + Element(15.0 / 91.0 * std::sqrt(39.0)) * sh_6_4 * y
    + Element(3.0 / 91.0 * std::sqrt(390.0)) * sh_6_5 * z
    + Element(3.0 / 91.0 * std::sqrt(390.0)) * sh_6_7 * x
    + Element(5.0 / 91.0 * std::sqrt(39.0)) * sh_6_9 * x;
const Element sh_7_6 = Element(-15.0 / 182.0 * std::sqrt(26.0)) * sh_6_4 * z
    + Element(12.0 / 91.0 * std::sqrt(65.0)) * sh_6_5 * y
    + Element(2.0 / 91.0 * std::sqrt(1365.0)) * sh_6_6 * x
    + Element(15.0 / 182.0 * std::sqrt(26.0)) * sh_6_8 * x;
const Element sh_7_7 = Element(-3.0 / 91.0 * std::sqrt(455.0)) * sh_6_5 * x
    + Element(1.0 / 13.0 * std::sqrt(195.0)) * sh_6_6 * y
    - Element(3.0 / 91.0 * std::sqrt(455.0)) * sh_6_7 * z;
const Element sh_7_8 = Element(-15.0 / 182.0 * std::sqrt(26.0)) * sh_6_4 * x
    + Element(2.0 / 91.0 * std::sqrt(1365.0)) * sh_6_6 * z
    + Element(12.0 / 91.0 * std::sqrt(65.0)) * sh_6_7 * y
    - Element(15.0 / 182.0 * std::sqrt(26.0)) * sh_6_8 * z;
const Element sh_7_9 = Element(-5.0 / 91.0 * std::sqrt(39.0)) * sh_6_3 * x
    - Element(3.0 / 91.0 * std::sqrt(390.0)) * sh_6_5 * x
    + Element(3.0 / 91.0 * std::sqrt(390.0)) * sh_6_7 * z
    + Element(15.0 / 91.0 * std::sqrt(39.0)) * sh_6_8 * y
    - Element(5.0 / 91.0 * std::sqrt(39.0)) * sh_6_9 * z;
const Element sh_7_10 = Element(-3.0 / 91.0 * std::sqrt(65.0)) * sh_6_10 * z
    - Element(3.0 / 91.0 * std::sqrt(65.0)) * sh_6_2 * x
    - Element(15.0 / 182.0 * std::sqrt(78.0)) * sh_6_4 * x
    + Element(15.0 / 182.0 * std::sqrt(78.0)) * sh_6_8 * z
    + Element(10.0 / 91.0 * std::sqrt(78.0)) * sh_6_9 * y;
const Element sh_7_11 = Element(-3.0 / 182.0 * std::sqrt(130.0)) * sh_6_1 * x
    + Element(3.0 / 91.0 * std::sqrt(715.0)) * sh_6_10 * y
    - Element(3.0 / 182.0 * std::sqrt(130.0)) * sh_6_11 * z
    - Element(5.0 / 182.0 * std::sqrt(858.0)) * sh_6_3 * x
    + Element(5.0 / 182.0 * std::sqrt(858.0)) * sh_6_9 * z;
const Element sh_7_12 = Element(-1.0 / 182.0 * std::sqrt(390.0)) * sh_6_0 * x
    + Element(3.0 / 91.0 * std::sqrt(715.0)) * sh_6_10 * z
    + Element(6.0 / 91.0 * std::sqrt(130.0)) * sh_6_11 * y
    - Element(1.0 / 182.0 * std::sqrt(390.0)) * sh_6_12 * z
    - Element(3.0 / 91.0 * std::sqrt(715.0)) * sh_6_2 * x;
const Element sh_7_13 = Element(-3.0 / 7.0 * std::sqrt(5.0)) * sh_6_1 * x
    + Element(3.0 / 7.0 * std::sqrt(5.0)) * sh_6_11 * z
    + Element(1.0 / 7.0 * std::sqrt(15.0)) * sh_6_12 * y;
const Element sh_7_14 = Element(1.0 / 14.0 * std::sqrt(210.0)) * (-sh_6_0 * x + sh_6_12 * z);

// Degree 8
const Element sh_8_0 = Element(1.0 / 4.0 * std::sqrt(17.0)) * (sh_7_0 * z + sh_7_14 * x);
const Element sh_8_1 = Element(1.0 / 8.0 * std::sqrt(17.0)) * sh_7_0 * y
    + Element(1.0 / 16.0 * std::sqrt(238.0)) * sh_7_1 * z
    + Element(1.0 / 16.0 * std::sqrt(238.0)) * sh_7_13 * x;
const Element sh_8_2 = Element(-1.0 / 240.0 * std::sqrt(510.0)) * sh_7_0 * z
    + Element(1.0 / 60.0 * std::sqrt(1785.0)) * sh_7_1 * y
    + Element(1.0 / 240.0 * std::sqrt(46410.0)) * sh_7_12 * x
    + Element(1.0 / 240.0 * std::sqrt(510.0)) * sh_7_14 * x
    + Element(1.0 / 240.0 * std::sqrt(46410.0)) * sh_7_2 * z;
const Element sh_8_3 = Element(1.0 / 80.0 * std::sqrt(2.0)) * (Element(-std::sqrt(85.0)) * sh_7_1 * z + Element(std::sqrt(2210.0)) * sh_7_11 * x + Element(std::sqrt(85.0)) * sh_7_13 * x + Element(std::sqrt(2210.0)) * sh_7_2 * y + Element(std::sqrt(2210.0)) * sh_7_3 * z);
const Element sh_8_4 = Element(1.0 / 40.0 * std::sqrt(935.0)) * sh_7_10 * x
    + Element(1.0 / 40.0 * std::sqrt(85.0)) * sh_7_12 * x
    - Element(1.0 / 40.0 * std::sqrt(85.0)) * sh_7_2 * z
    + Element(1.0 / 10.0 * std::sqrt(85.0)) * sh_7_3 * y
    + Element(1.0 / 40.0 * std::sqrt(935.0)) * sh_7_4 * z;
const Element sh_8_5 = Element(1.0 / 48.0 * std::sqrt(2.0)) * (Element(std::sqrt(102.0)) * sh_7_11 * x - Element(std::sqrt(102.0)) * sh_7_3 * z + Element(std::sqrt(1122.0)) * sh_7_4 * y + Element(std::sqrt(561.0)) * sh_7_5 * z + Element(std::sqrt(561.0)) * sh_7_9 * x);
const Element sh_8_6 = Element(1.0 / 16.0 * std::sqrt(34.0)) * sh_7_10 * x
    - Element(1.0 / 16.0 * std::sqrt(34.0)) * sh_7_4 * z
    + Element(1.0 / 4.0 * std::sqrt(17.0)) * sh_7_5 * y
    + Element(1.0 / 16.0 * std::sqrt(102.0)) * sh_7_6 * z
    + Element(1.0 / 16.0 * std::sqrt(102.0)) * sh_7_8 * x;
const Element sh_8_7 = Element(-1.0 / 80.0 * std::sqrt(1190.0)) * sh_7_5 * z
    + Element(1.0 / 40.0 * std::sqrt(1785.0)) * sh_7_6 * y
    + Element(1.0 / 20.0 * std::sqrt(255.0)) * sh_7_7 * x
    + Element(1.0 / 80.0 * std::sqrt(1190.0)) * sh_7_9 * x;
const Element sh_8_8 = Element(-1.0 / 60.0 * std::sqrt(1785.0)) * sh_7_6 * x
    + Element(1.0 / 15.0 * std::sqrt(255.0)) * sh_7_7 * y
    - Element(1.0 / 60.0 * std::sqrt(1785.0)) * sh_7_8 * z;
const Element sh_8_9 = Element(-1.0 / 80.0 * std::sqrt(1190.0)) * sh_7_5 * x
    + Element(1.0 / 20.0 * std::sqrt(255.0)) * sh_7_7 * z
    + Element(1.0 / 40.0 * std::sqrt(1785.0)) * sh_7_8 * y
    - Element(1.0 / 80.0 * std::sqrt(1190.0)) * sh_7_9 * z;
const Element sh_8_10 = Element(-1.0 / 16.0 * std::sqrt(34.0)) * sh_7_10 * z
    - Element(1.0 / 16.0 * std::sqrt(34.0)) * sh_7_4 * x
    - Element(1.0 / 16.0 * std::sqrt(102.0)) * sh_7_6 * x
    + Element(1.0 / 16.0 * std::sqrt(102.0)) * sh_7_8 * z
    + Element(1.0 / 4.0 * std::sqrt(17.0)) * sh_7_9 * y;
const Element sh_8_11 = Element(1.0 / 48.0 * std::sqrt(2.0)) * (Element(std::sqrt(1122.0)) * sh_7_10 * y - Element(std::sqrt(102.0)) * sh_7_11 * z - Element(std::sqrt(102.0)) * sh_7_3 * x - Element(std::sqrt(561.0)) * sh_7_5 * x + Element(std::sqrt(561.0)) * sh_7_9 * z);
const Element sh_8_12 = Element(1.0 / 40.0 * std::sqrt(935.0)) * sh_7_10 * z
    + Element(1.0 / 10.0 * std::sqrt(85.0)) * sh_7_11 * y
    - Element(1.0 / 40.0 * std::sqrt(85.0)) * sh_7_12 * z
    - Element(1.0 / 40.0 * std::sqrt(85.0)) * sh_7_2 * x
    - Element(1.0 / 40.0 * std::sqrt(935.0)) * sh_7_4 * x;
const Element sh_8_13 = Element(1.0 / 80.0 * std::sqrt(2.0)) * (Element(-std::sqrt(85.0)) * sh_7_1 * x + Element(std::sqrt(2210.0)) * sh_7_11 * z + Element(std::sqrt(2210.0)) * sh_7_12 * y - Element(std::sqrt(85.0)) * sh_7_13 * z - Element(std::sqrt(2210.0)) * sh_7_3 * x);
const Element sh_8_14 = Element(-1.0 / 240.0 * std::sqrt(510.0)) * sh_7_0 * x
    + Element(1.0 / 240.0 * std::sqrt(46410.0)) * sh_7_12 * z
    + Element(1.0 / 60.0 * std::sqrt(1785.0)) * sh_7_13 * y
    - Element(1.0 / 240.0 * std::sqrt(510.0)) * sh_7_14 * z
    - Element(1.0 / 240.0 * std::sqrt(46410.0)) * sh_7_2 * x;
const Element sh_8_15 = Element(-1.0 / 16.0 * std::sqrt(238.0)) * sh_7_1 * x
    + Element(1.0 / 16.0 * std::sqrt(238.0)) * sh_7_13 * z
    + Element(1.0 / 8.0 * std::sqrt(17.0)) * sh_7_14 * y;
const Element sh_8_16 = Element(1.0 / 4.0 * std::sqrt(17.0)) * (-sh_7_0 * x + sh_7_14 * z);

// Degree 9
const Element sh_9_0 = Element(1.0 / 6.0 * std::sqrt(38.0)) * (sh_8_0 * z + sh_8_16 * x);
const Element sh_9_1 = Element(1.0 / 9.0 * std::sqrt(19.0)) * (sh_8_0 * y + Element(2.0) * sh_8_1 * z + Element(2.0) * sh_8_15 * x);
const Element sh_9_2 = Element(-1.0 / 306.0 * std::sqrt(646.0)) * sh_8_0 * z
    + Element(4.0 / 153.0 * std::sqrt(646.0)) * sh_8_1 * y
    + Element(2.0 / 153.0 * std::sqrt(4845.0)) * sh_8_14 * x
    + Element(1.0 / 306.0 * std::sqrt(646.0)) * sh_8_16 * x
    + Element(2.0 / 153.0 * std::sqrt(4845.0)) * sh_8_2 * z;
const Element sh_9_3 = Element(-1.0 / 306.0 * std::sqrt(1938.0)) * sh_8_1 * z
    + Element(1.0 / 306.0 * std::sqrt(67830.0)) * sh_8_13 * x
    + Element(1.0 / 306.0 * std::sqrt(1938.0)) * sh_8_15 * x
    + Element(1.0 / 51.0 * std::sqrt(1615.0)) * sh_8_2 * y
    + Element(1.0 / 306.0 * std::sqrt(67830.0)) * sh_8_3 * z;
const Element sh_9_4 = Element(1.0 / 306.0 * std::sqrt(58786.0)) * sh_8_12 * x
    + Element(1.0 / 153.0 * std::sqrt(969.0)) * sh_8_14 * x
    - Element(1.0 / 153.0 * std::sqrt(969.0)) * sh_8_2 * z
    + Element(2.0 / 153.0 * std::sqrt(4522.0)) * sh_8_3 * y
    + Element(1.0 / 306.0 * std::sqrt(58786.0)) * sh_8_4 * z;
const Element sh_9_5 = Element(1.0 / 153.0 * std::sqrt(12597.0)) * sh_8_11 * x
    + Element(1.0 / 153.0 * std::sqrt(1615.0)) * sh_8_13 * x
    - Element(1.0 / 153.0 * std::sqrt(1615.0)) * sh_8_3 * z
    + Element(1.0 / 153.0 * std::sqrt(20995.0)) * sh_8_4 * y
    + Element(1.0 / 153.0 * std::sqrt(12597.0)) * sh_8_5 * z;
const Element sh_9_6 = Element(1.0 / 153.0 * std::sqrt(10659.0)) * sh_8_10 * x
    + Element(1.0 / 306.0 * std::sqrt(9690.0)) * sh_8_12 * x
    - Element(1.0 / 306.0 * std::sqrt(9690.0)) * sh_8_4 * z
    + Element(2.0 / 51.0 * std::sqrt(646.0)) * sh_8_5 * y
    + Element(1.0 / 153.0 * std::sqrt(10659.0)) * sh_8_6 * z;
const Element sh_9_7 = Element(1.0 / 306.0 * std::sqrt(13566.0)) * sh_8_11 * x
    - Element(1.0 / 306.0 * std::sqrt(13566.0)) * sh_8_5 * z
    + Element(1.0 / 153.0 * std::sqrt(24871.0)) * sh_8_6 * y
    + Element(1.0 / 306.0 * std::sqrt(35530.0)) * sh_8_7 * z
    + Element(1.0 / 306.0 * std::sqrt(35530.0)) * sh_8_9 * x;
const Element sh_9_8 = Element(1.0 / 153.0 * std::sqrt(4522.0)) * sh_8_10 * x
    - Element(1.0 / 153.0 * std::sqrt(4522.0)) * sh_8_6 * z
    + Element(4.0 / 153.0 * std::sqrt(1615.0)) * sh_8_7 * y
    + Element(1.0 / 51.0 * std::sqrt(1615.0)) * sh_8_8 * x;
const Element sh_9_9 = Element(1.0 / 51.0 * std::sqrt(323.0)) * (Element(-2.0) * sh_8_7 * x + Element(3.0) * sh_8_8 * y - Element(2.0) * sh_8_9 * z);
const Element sh_9_10 = Element(-1.0 / 153.0 * std::sqrt(4522.0)) * sh_8_10 * z
    - Element(1.0 / 153.0 * std::sqrt(4522.0)) * sh_8_6 * x
    + Element(1.0 / 51.0 * std::sqrt(1615.0)) * sh_8_8 * z
    + Element(4.0 / 153.0 * std::sqrt(1615.0)) * sh_8_9 * y;
const Element sh_9_11 = Element(1.0 / 153.0 * std::sqrt(24871.0)) * sh_8_10 * y
    - Element(1.0 / 306.0 * std::sqrt(13566.0)) * sh_8_11 * z
    - Element(1.0 / 306.0 * std::sqrt(13566.0)) * sh_8_5 * x
    - Element(1.0 / 306.0 * std::sqrt(35530.0)) * sh_8_7 * x
    + Element(1.0 / 306.0 * std::sqrt(35530.0)) * sh_8_9 * z;
const Element sh_9_12 = Element(1.0 / 153.0 * std::sqrt(10659.0)) * sh_8_10 * z
    + Element(2.0 / 51.0 * std::sqrt(646.0)) * sh_8_11 * y
    - Element(1.0 / 306.0 * std::sqrt(9690.0)) * sh_8_12 * z
    - Element(1.0 / 306.0 * std::sqrt(9690.0)) * sh_8_4 * x
    - Element(1.0 / 153.0 * std::sqrt(10659.0)) * sh_8_6 * x;
const Element sh_9_13 = Element(1.0 / 153.0 * std::sqrt(12597.0)) * sh_8_11 * z
    + Element(1.0 / 153.0 * std::sqrt(20995.0)) * sh_8_12 * y
    - Element(1.0 / 153.0 * std::sqrt(1615.0)) * sh_8_13 * z
    - Element(1.0 / 153.0 * std::sqrt(1615.0)) * sh_8_3 * x
    - Element(1.0 / 153.0 * std::sqrt(12597.0)) * sh_8_5 * x;
const Element sh_9_14 = Element(1.0 / 306.0 * std::sqrt(58786.0)) * sh_8_12 * z
    + Element(2.0 / 153.0 * std::sqrt(4522.0)) * sh_8_13 * y
    - Element(1.0 / 153.0 * std::sqrt(969.0)) * sh_8_14 * z
    - Element(1.0 / 153.0 * std::sqrt(969.0)) * sh_8_2 * x
    - Element(1.0 / 306.0 * std::sqrt(58786.0)) * sh_8_4 * x;
const Element sh_9_15 = Element(-1.0 / 306.0 * std::sqrt(1938.0)) * sh_8_1 * x
    + Element(1.0 / 306.0 * std::sqrt(67830.0)) * sh_8_13 * z
    + Element(1.0 / 51.0 * std::sqrt(1615.0)) * sh_8_14 * y
    - Element(1.0 / 306.0 * std::sqrt(1938.0)) * sh_8_15 * z
    - Element(1.0 / 306.0 * std::sqrt(67830.0)) * sh_8_3 * x;
const Element sh_9_16 = Element(-1.0 / 306.0 * std::sqrt(646.0)) * sh_8_0 * x
    + Element(2.0 / 153.0 * std::sqrt(4845.0)) * sh_8_14 * z
    + Element(4.0 / 153.0 * std::sqrt(646.0)) * sh_8_15 * y
    - Element(1.0 / 306.0 * std::sqrt(646.0)) * sh_8_16 * z
    - Element(2.0 / 153.0 * std::sqrt(4845.0)) * sh_8_2 * x;
const Element sh_9_17 = Element(1.0 / 9.0 * std::sqrt(19.0)) * (Element(-2.0) * sh_8_1 * x + Element(2.0) * sh_8_15 * z + sh_8_16 * y);
const Element sh_9_18 = Element(1.0 / 6.0 * std::sqrt(38.0)) * (-sh_8_0 * x + sh_8_16 * z);

// Degree 10
const Element sh_10_0 = Element(1.0 / 10.0 * std::sqrt(105.0)) * (sh_9_0 * z + sh_9_18 * x);
const Element sh_10_1 = Element(1.0 / 10.0 * std::sqrt(21.0)) * sh_9_0 * y
    + Element(3.0 / 20.0 * std::sqrt(42.0)) * sh_9_1 * z
    + Element(3.0 / 20.0 * std::sqrt(42.0)) * sh_9_17 * x;
const Element sh_10_2 = Element(-1.0 / 380.0 * std::sqrt(798.0)) * sh_9_0 * z
    + Element(3.0 / 95.0 * std::sqrt(399.0)) * sh_9_1 * y
    + Element(3.0 / 380.0 * std::sqrt(13566.0)) * sh_9_16 * x
    + Element(1.0 / 380.0 * std::sqrt(798.0)) * sh_9_18 * x
    + Element(3.0 / 380.0 * std::sqrt(13566.0)) * sh_9_2 * z;
const Element sh_10_3 = Element(-3.0 / 380.0 * std::sqrt(266.0)) * sh_9_1 * z
    + Element(1.0 / 95.0 * std::sqrt(6783.0)) * sh_9_15 * x
    + Element(3.0 / 380.0 * std::sqrt(266.0)) * sh_9_17 * x
    + Element(3.0 / 190.0 * std::sqrt(2261.0)) * sh_9_2 * y
    + Element(1.0 / 95.0 * std::sqrt(6783.0)) * sh_9_3 * z;
const Element sh_10_4 = Element(3.0 / 95.0 * std::sqrt(665.0)) * sh_9_14 * x
    + Element(3.0 / 190.0 * std::sqrt(133.0)) * sh_9_16 * x
    - Element(3.0 / 190.0 * std::sqrt(133.0)) * sh_9_2 * z
    + Element(4.0 / 95.0 * std::sqrt(399.0)) * sh_9_3 * y
    + Element(3.0 / 95.0 * std::sqrt(665.0)) * sh_9_4 * z;
const Element sh_10_5 = Element(21.0 / 380.0 * std::sqrt(190.0)) * sh_9_13 * x
    + Element(1.0 / 190.0 * std::sqrt(1995.0)) * sh_9_15 * x
    - Element(1.0 / 190.0 * std::sqrt(1995.0)) * sh_9_3 * z
    + Element(3.0 / 38.0 * std::sqrt(133.0)) * sh_9_4 * y
    + Element(21.0 / 380.0 * std::sqrt(190.0)) * sh_9_5 * z;
const Element sh_10_6 = Element(7.0 / 380.0 * std::sqrt(1482.0)) * sh_9_12 * x
    + Element(3.0 / 380.0 * std::sqrt(1330.0)) * sh_9_14 * x
    - Element(3.0 / 380.0 * std::sqrt(1330.0)) * sh_9_4 * z
    + Element(21.0 / 95.0 * std::sqrt(19.0)) * sh_9_5 * y
    + Element(7.0 / 380.0 * std::sqrt(1482.0)) * sh_9_6 * z;
const Element sh_10_7 = Element(3.0 / 190.0 * std::sqrt(1729.0)) * sh_9_11 * x
    + Element(21.0 / 380.0 * std::sqrt(38.0)) * sh_9_13 * x
    - Element(21.0 / 380.0 * std::sqrt(38.0)) * sh_9_5 * z
    + Element(7.0 / 190.0 * std::sqrt(741.0)) * sh_9_6 * y
    + Element(3.0 / 190.0 * std::sqrt(1729.0)) * sh_9_7 * z;
const Element sh_10_8 = Element(3.0 / 190.0 * std::sqrt(1463.0)) * sh_9_10 * x
    + Element(7.0 / 190.0 * std::sqrt(114.0)) * sh_9_12 * x
    - Element(7.0 / 190.0 * std::sqrt(114.0)) * sh_9_6 * z
    + Element(6.0 / 95.0 * std::sqrt(266.0)) * sh_9_7 * y
    + Element(3.0 / 190.0 * std::sqrt(1463.0)) * sh_9_8 * z;
const Element sh_10_9 = Element(3.0 / 190.0 * std::sqrt(798.0)) * sh_9_11 * x
    - Element(3.0 / 190.0 * std::sqrt(798.0)) * sh_9_7 * z
    + Element(3.0 / 190.0 * std::sqrt(4389.0)) * sh_9_8 * y
    + Element(1.0 / 190.0 * std::sqrt(21945.0)) * sh_9_9 * x;
const Element sh_10_10 = Element(-3.0 / 190.0 * std::sqrt(1995.0)) * sh_9_10 * z
    - Element(3.0 / 190.0 * std::sqrt(1995.0)) * sh_9_8 * x
    + Element(1.0 / 19.0 * std::sqrt(399.0)) * sh_9_9 * y;
const Element sh_10_11 = Element(3.0 / 190.0 * std::sqrt(4389.0)) * sh_9_10 * y
    - Element(3.0 / 190.0 * std::sqrt(798.0)) * sh_9_11 * z
    - Element(3.0 / 190.0 * std::sqrt(798.0)) * sh_9_7 * x
    + Element(1.0 / 190.0 * std::sqrt(21945.0)) * sh_9_9 * z;
const Element sh_10_12 = Element(3.0 / 190.0 * std::sqrt(1463.0)) * sh_9_10 * z
    + Element(6.0 / 95.0 * std::sqrt(266.0)) * sh_9_11 * y
    - Element(7.0 / 190.0 * std::sqrt(114.0)) * sh_9_12 * z
    - Element(7.0 / 190.0 * std::sqrt(114.0)) * sh_9_6 * x
    - Element(3.0 / 190.0 * std::sqrt(1463.0)) * sh_9_8 * x;
const Element sh_10_13 = Element(3.0 / 190.0 * std::sqrt(1729.0)) * sh_9_11 * z
    + Element(7.0 / 190.0 * std::sqrt(741.0)) * sh_9_12 * y
    - Element(21.0 / 380.0 * std::sqrt(38.0)) * sh_9_13 * z
    - Element(21.0 / 380.0 * std::sqrt(38.0)) * sh_9_5 * x
    - Element(3.0 / 190.0 * std::sqrt(1729.0)) * sh_9_7 * x;
const Element sh_10_14 = Element(7.0 / 380.0 * std::sqrt(1482.0)) * sh_9_12 * z
    + Element(21.0 / 95.0 * std::sqrt(19.0)) * sh_9_13 * y
    - Element(3.0 / 380.0 * std::sqrt(1330.0)) * sh_9_14 * z
    - Element(3.0 / 380.0 * std::sqrt(1330.0)) * sh_9_4 * x
    - Element(7.0 / 380.0 * std::sqrt(1482.0)) * sh_9_6 * x;
const Element sh_10_15 = Element(21.0 / 380.0 * std::sqrt(190.0)) * sh_9_13 * z
    + Element(3.0 / 38.0 * std::sqrt(133.0)) * sh_9_14 * y
    - Element(1.0 / 190.0 * std::sqrt(1995.0)) * sh_9_15 * z
    - Element(1.0 / 190.0 * std::sqrt(1995.0)) * sh_9_3 * x
    - Element(21.0 / 380.0 * std::sqrt(190.0)) * sh_9_5 * x;
const Element sh_10_16 = Element(3.0 / 95.0 * std::sqrt(665.0)) * sh_9_14 * z
    + Element(4.0 / 95.0 * std::sqrt(399.0)) * sh_9_15 * y
    - Element(3.0 / 190.0 * std::sqrt(133.0)) * sh_9_16 * z
    - Element(3.0 / 190.0 * std::sqrt(133.0)) * sh_9_2 * x
    - Element(3.0 / 95.0 * std::sqrt(665.0)) * sh_9_4 * x;
const Element sh_10_17 = Element(-3.0 / 380.0 * std::sqrt(266.0)) * sh_9_1 * x
    + Element(1.0 / 95.0 * std::sqrt(6783.0)) * sh_9_15 * z
    + Element(3.0 / 190.0 * std::sqrt(2261.0)) * sh_9_16 * y
    - Element(3.0 / 380.0 * std::sqrt(266.0)) * sh_9_17 * z
    - Element(1.0 / 95.0 * std::sqrt(6783.0)) * sh_9_3 * x;
const Element sh_10_18 = Element(-1.0 / 380.0 * std::sqrt(798.0)) * sh_9_0 * x
    + Element(3.0 / 380.0 * std::sqrt(13566.0)) * sh_9_16 * z
    + Element(3.0 / 95.0 * std::sqrt(399.0)) * sh_9_17 * y
    - Element(1.0 / 380.0 * std::sqrt(798.0)) * sh_9_18 * z
    - Element(3.0 / 380.0 * std::sqrt(13566.0)) * sh_9_2 * x;
const Element sh_10_19 = Element(-3.0 / 20.0 * std::sqrt(42.0)) * sh_9_1 * x
    + Element(3.0 / 20.0 * std::sqrt(42.0)) * sh_9_17 * z
    + Element(1.0 / 10.0 * std::sqrt(21.0)) * sh_9_18 * y;
const Element sh_10_20 = Element(1.0 / 10.0 * std::sqrt(105.0)) * (-sh_9_0 * x + sh_9_18 * z);

// Degree 11
const Element sh_11_0 = Element(1.0 / 22.0 * std::sqrt(506.0)) * (sh_10_0 * z + sh_10_20 * x);
const Element sh_11_1 = Element(1.0 / 11.0 * std::sqrt(23.0)) * sh_10_0 * y
    + Element(1.0 / 11.0 * std::sqrt(115.0)) * sh_10_1 * z
    + Element(1.0 / 11.0 * std::sqrt(115.0)) * sh_10_19 * x;
const Element sh_11_2 = Element(-1.0 / 462.0 * std::sqrt(966.0)) * sh_10_0 * z
    + Element(2.0 / 231.0 * std::sqrt(4830.0)) * sh_10_1 * y
    + Element(1.0 / 231.0 * std::sqrt(45885.0)) * sh_10_18 * x
    + Element(1.0 / 231.0 * std::sqrt(45885.0)) * sh_10_2 * z
    + Element(1.0 / 462.0 * std::sqrt(966.0)) * sh_10_20 * x;
const Element sh_11_3 = Element(-1.0 / 154.0 * std::sqrt(322.0)) * sh_10_1 * z
    + Element(1.0 / 154.0 * std::sqrt(18354.0)) * sh_10_17 * x
    + Element(1.0 / 154.0 * std::sqrt(322.0)) * sh_10_19 * x
    + Element(1.0 / 77.0 * std::sqrt(3059.0)) * sh_10_2 * y
    + Element(1.0 / 154.0 * std::sqrt(18354.0)) * sh_10_3 * z;
const Element sh_11_4 = Element(1.0 / 154.0 * std::sqrt(16422.0)) * sh_10_16 * x
    + Element(1.0 / 77.0 * std::sqrt(161.0)) * sh_10_18 * x
    - Element(1.0 / 77.0 * std::sqrt(161.0)) * sh_10_2 * z
    + Element(2.0 / 77.0 * std::sqrt(966.0)) * sh_10_3 * y
    + Element(1.0 / 154.0 * std::sqrt(16422.0)) * sh_10_4 * z;
const Element sh_11_5 = Element(2.0 / 231.0 * std::sqrt(8211.0)) * sh_10_15 * x
    + Element(1.0 / 231.0 * std::sqrt(2415.0)) * sh_10_17 * x
    - Element(1.0 / 231.0 * std::sqrt(2415.0)) * sh_10_3 * z
    + Element(1.0 / 231.0 * std::sqrt(41055.0)) * sh_10_4 * y
    + Element(2.0 / 231.0 * std::sqrt(8211.0)) * sh_10_5 * z;
const Element sh_11_6 = Element(2.0 / 77.0 * std::sqrt(805.0)) * sh_10_14 * x
    + Element(1.0 / 154.0 * std::sqrt(1610.0)) * sh_10_16 * x
    - Element(1.0 / 154.0 * std::sqrt(1610.0)) * sh_10_4 * z
    + Element(4.0 / 77.0 * std::sqrt(322.0)) * sh_10_5 * y
    + Element(2.0 / 77.0 * std::sqrt(805.0)) * sh_10_6 * z;
const Element sh_11_7 = Element(1.0 / 22.0 * std::sqrt(230.0)) * sh_10_13 * x
    + Element(1.0 / 22.0 * std::sqrt(46.0)) * sh_10_15 * x
    - Element(1.0 / 22.0 * std::sqrt(46.0)) * sh_10_5 * z
    + Element(1.0 / 11.0 * std::sqrt(115.0)) * sh_10_6 * y
    + Element(1.0 / 22.0 * std::sqrt(230.0)) * sh_10_7 * z;
const Element sh_11_8 = Element(1.0 / 66.0 * std::sqrt(1794.0)) * sh_10_12 * x
    + Element(1.0 / 33.0 * std::sqrt(138.0)) * sh_10_14 * x
    - Element(1.0 / 33.0 * std::sqrt(138.0)) * sh_10_6 * z
    + Element(4.0 / 33.0 * std::sqrt(69.0)) * sh_10_7 * y
    + Element(1.0 / 66.0 * std::sqrt(1794.0)) * sh_10_8 * z;
const Element sh_11_9 = Element(1.0 / 77.0 * std::sqrt(2093.0)) * sh_10_11 * x
    + Element(1.0 / 77.0 * std::sqrt(966.0)) * sh_10_13 * x
    - Element(1.0 / 77.0 * std::sqrt(966.0)) * sh_10_7 * z
    + Element(1.0 / 77.0 * std::sqrt(6279.0)) * sh_10_8 * y
    + Element(1.0 / 77.0 * std::sqrt(2093.0)) * sh_10_9 * z;
const Element sh_11_10 = Element(1.0 / 77.0 * std::sqrt(3542.0)) * sh_10_10 * x
    + Element(1.0 / 154.0 * std::sqrt(4830.0)) * sh_10_12 * x
    - Element(1.0 / 154.0 * std::sqrt(4830.0)) * sh_10_8 * z
    + Element(2.0 / 77.0 * std::sqrt(1610.0)) * sh_10_9 * y;
const Element sh_11_11 = Element(1.0 / 21.0 * std::sqrt(483.0)) * sh_10_10 * y
    - Element(1.0 / 231.0 * std::sqrt(26565.0)) * sh_10_11 * z
    - Element(1.0 / 231.0 * std::sqrt(26565.0)) * sh_10_9 * x;
const Element sh_11_12 = Element(1.0 / 77.0 * std::sqrt(3542.0)) * sh_10_10 * z
    + Element(2.0 / 77.0 * std::sqrt(1610.0)) * sh_10_11 * y
    - Element(1.0 / 154.0 * std::sqrt(4830.0)) * sh_10_12 * z
    - Element(1.0 / 154.0 * std::sqrt(4830.0)) * sh_10_8 * x;
const Element sh_11_13 = Element(1.0 / 77.0 * std::sqrt(2093.0)) * sh_10_11 * z
    + Element(1.0 / 77.0 * std::sqrt(6279.0)) * sh_10_12 * y
    - Element(1.0 / 77.0 * std::sqrt(966.0)) * sh_10_13 * z
    - Element(1.0 / 77.0 * std::sqrt(966.0)) * sh_10_7 * x
    - Element(1.0 / 77.0 * std::sqrt(2093.0)) * sh_10_9 * x;
const Element sh_11_14 = Element(1.0 / 66.0 * std::sqrt(1794.0)) * sh_10_12 * z
    + Element(4.0 / 33.0 * std::sqrt(69.0)) * sh_10_13 * y
    - Element(1.0 / 33.0 * std::sqrt(138.0)) * sh_10_14 * z
    - Element(1.0 / 33.0 * std::sqrt(138.0)) * sh_10_6 * x
    - Element(1.0 / 66.0 * std::sqrt(1794.0)) * sh_10_8 * x;
const Element sh_11_15 = Element(1.0 / 22.0 * std::sqrt(230.0)) * sh_10_13 * z
    + Element(1.0 / 11.0 * std::sqrt(115.0)) * sh_10_14 * y
    - Element(1.0 / 22.0 * std::sqrt(46.0)) * sh_10_15 * z
    - Element(1.0 / 22.0 * std::sqrt(46.0)) * sh_10_5 * x
    - Element(1.0 / 22.0 * std::sqrt(230.0)) * sh_10_7 * x;
const Element sh_11_16 = Element(2.0 / 77.0 * std::sqrt(805.0)) * sh_10_14 * z
    + Element(4.0 / 77.0 * std::sqrt(322.0)) * sh_10_15 * y
    - Element(1.0 / 154.0 * std::sqrt(1610.0)) * sh_10_16 * z
    - Element(1.0 / 154.0 * std::sqrt(1610.0)) * sh_10_4 * x
    - Element(2.0 / 77.0 * std::sqrt(805.0)) * sh_10_6 * x;
const Element sh_11_17 = Element(2.0 / 231.0 * std::sqrt(8211.0)) * sh_10_15 * z
    + Element(1.0 / 231.0 * std::sqrt(41055.0)) * sh_10_16 * y
    - Element(1.0 / 231.0 * std::sqrt(2415.0)) * sh_10_17 * z
    - Element(1.0 / 231.0 * std::sqrt(2415.0)) * sh_10_3 * x
    - Element(2.0 / 231.0 * std::sqrt(8211.0)) * sh_10_5 * x;
const Element sh_11_18 = Element(1.0 / 154.0 * std::sqrt(16422.0)) * sh_10_16 * z
    + Element(2.0 / 77.0 * std::sqrt(966.0)) * sh_10_17 * y
    - Element(1.0 / 77.0 * std::sqrt(161.0)) * sh_10_18 * z
    - Element(1.0 / 77.0 * std::sqrt(161.0)) * sh_10_2 * x
    - Element(1.0 / 154.0 * std::sqrt(16422.0)) * sh_10_4 * x;
const Element sh_11_19 = Element(-1.0 / 154.0 * std::sqrt(322.0)) * sh_10_1 * x
    + Element(1.0 / 154.0 * std::sqrt(18354.0)) * sh_10_17 * z
    + Element(1.0 / 77.0 * std::sqrt(3059.0)) * sh_10_18 * y
    - Element(1.0 / 154.0 * std::sqrt(322.0)) * sh_10_19 * z
    - Element(1.0 / 154.0 * std::sqrt(18354.0)) * sh_10_3 * x;
const Element sh_11_20 = Element(-1.0 / 462.0 * std::sqrt(966.0)) * sh_10_0 * x
    + Element(1.0 / 231.0 * std::sqrt(45885.0)) * sh_10_18 * z
    + Element(2.0 / 231.0 * std::sqrt(4830.0)) * sh_10_19 * y
    - Element(1.0 / 231.0 * std::sqrt(45885.0)) * sh_10_2 * x
    - Element(1.0 / 462.0 * std::sqrt(966.0)) * sh_10_20 * z;
const Element sh_11_21 = Element(-1.0 / 11.0 * std::sqrt(115.0)) * sh_10_1 * x
    + Element(1.0 / 11.0 * std::sqrt(115.0)) * sh_10_19 * z
    + Element(1.0 / 11.0 * std::sqrt(23.0)) * sh_10_20 * y;
const Element sh_11_22 = Element(1.0 / 22.0 * std::sqrt(506.0)) * (-sh_10_0 * x + sh_10_20 * z);

// Degree 12
const Element sh_12_0 = Element(5.0 / 12.0 * std::sqrt(6.0)) * (sh_11_0 * z + sh_11_22 * x);
const Element sh_12_1 = Element(5.0 / 12.0) * sh_11_0 * y
    + Element(5.0 / 24.0 * std::sqrt(22.0)) * sh_11_1 * z
    + Element(5.0 / 24.0 * std::sqrt(22.0)) * sh_11_21 * x;
const Element sh_12_2 = Element(-5.0 / 552.0 * std::sqrt(46.0)) * sh_11_0 * z
    + Element(5.0 / 138.0 * std::sqrt(253.0)) * sh_11_1 * y
    + Element(5.0 / 552.0 * std::sqrt(10626.0)) * sh_11_2 * z
    + Element(5.0 / 552.0 * std::sqrt(10626.0)) * sh_11_20 * x
    + Element(5.0 / 552.0 * std::sqrt(46.0)) * sh_11_22 * x;
const Element sh_12_3 = Element(-5.0 / 552.0 * std::sqrt(138.0)) * sh_11_1 * z
    + Element(5.0 / 276.0 * std::sqrt(2415.0)) * sh_11_19 * x
    + Element(5.0 / 92.0 * std::sqrt(161.0)) * sh_11_2 * y
    + Element(5.0 / 552.0 * std::sqrt(138.0)) * sh_11_21 * x
    + Element(5.0 / 276.0 * std::sqrt(2415.0)) * sh_11_3 * z;
const Element sh_12_4 = Element(5.0 / 276.0 * std::sqrt(2185.0)) * sh_11_18 * x
    - Element(5.0 / 276.0 * std::sqrt(69.0)) * sh_11_2 * z
    + Element(5.0 / 276.0 * std::sqrt(69.0)) * sh_11_20 * x
    + Element(5.0 / 69.0 * std::sqrt(115.0)) * sh_11_3 * y
    + Element(5.0 / 276.0 * std::sqrt(2185.0)) * sh_11_4 * z;
const Element sh_12_5 = Element(5.0 / 184.0 * std::sqrt(874.0)) * sh_11_17 * x
    + Element(5.0 / 276.0 * std::sqrt(115.0)) * sh_11_19 * x
    - Element(5.0 / 276.0 * std::sqrt(115.0)) * sh_11_3 * z
    + Element(5.0 / 276.0 * std::sqrt(2185.0)) * sh_11_4 * y
    + Element(5.0 / 184.0 * std::sqrt(874.0)) * sh_11_5 * z;
const Element sh_12_6 = Element(5.0 / 552.0 * std::sqrt(3.0)) * (Element(std::sqrt(2346.0)) * sh_11_16 * x + Element(std::sqrt(230.0)) * sh_11_18 * x - Element(std::sqrt(230.0)) * sh_11_4 * z + Element(12.0 * std::sqrt(23.0)) * sh_11_5 * y + Element(std::sqrt(2346.0)) * sh_11_6 * z);
const Element sh_12_7 = Element(5.0 / 138.0 * std::sqrt(391.0)) * sh_11_15 * x
    + Element(5.0 / 552.0 * std::sqrt(966.0)) * sh_11_17 * x
    - Element(5.0 / 552.0 * std::sqrt(966.0)) * sh_11_5 * z
    + Element(5.0 / 276.0 * std::sqrt(2737.0)) * sh_11_6 * y
    + Element(5.0 / 138.0 * std::sqrt(391.0)) * sh_11_7 * z;
const Element sh_12_8 = Element(5.0 / 138.0 * std::sqrt(345.0)) * sh_11_14 * x
    + Element(5.0 / 276.0 * std::sqrt(322.0)) * sh_11_16 * x
    - Element(5.0 / 276.0 * std::sqrt(322.0)) * sh_11_6 * z
    + Element(10.0 / 69.0 * std::sqrt(46.0)) * sh_11_7 * y
    + Element(5.0 / 138.0 * std::sqrt(345.0)) * sh_11_8 * z;
const Element sh_12_9 = Element(5.0 / 552.0 * std::sqrt(4830.0)) * sh_11_13 * x
    + Element(5.0 / 92.0 * std::sqrt(46.0)) * sh_11_15 * x
    - Element(5.0 / 92.0 * std::sqrt(46.0)) * sh_11_7 * z
    + Element(5.0 / 92.0 * std::sqrt(345.0)) * sh_11_8 * y
    + Element(5.0 / 552.0 * std::sqrt(4830.0)) * sh_11_9 * z;
const Element sh_12_10 = Element(5.0 / 552.0 * std::sqrt(4186.0)) * sh_11_10 * z
    + Element(5.0 / 552.0 * std::sqrt(4186.0)) * sh_11_12 * x
    + Element(5.0 / 184.0 * std::sqrt(230.0)) * sh_11_14 * x
    - Element(5.0 / 184.0 * std::sqrt(230.0)) * sh_11_8 * z
    + Element(5.0 / 138.0 * std::sqrt(805.0)) * sh_11_9 * y;
const Element sh_12_11 = Element(5.0 / 276.0 * std::sqrt(3289.0)) * sh_11_10 * y
    + Element(5.0 / 276.0 * std::sqrt(1794.0)) * sh_11_11 * x
    + Element(5.0 / 552.0 * std::sqrt(2530.0)) * sh_11_13 * x
    - Element(5.0 / 552.0 * std::sqrt(2530.0)) * sh_11_9 * z;
const Element sh_12_12 = Element(-5.0 / 276.0 * std::sqrt(1518.0)) * sh_11_10 * x
    + Element(5.0 / 23.0 * std::sqrt(23.0)) * sh_11_11 * y
    - Element(5.0 / 276.0 * std::sqrt(1518.0)) * sh_11_12 * z;
const Element sh_12_13 = Element(5.0 / 276.0 * std::sqrt(1794.0)) * sh_11_11 * z
    + Element(5.0 / 276.0 * std::sqrt(3289.0)) * sh_11_12 * y
    - Element(5.0 / 552.0 * std::sqrt(2530.0)) * sh_11_13 * z
    - Element(5.0 / 552.0 * std::sqrt(2530.0)) * sh_11_9 * x;
const Element sh_12_14 = Element(-5.0 / 552.0 * std::sqrt(4186.0)) * sh_11_10 * x
    + Element(5.0 / 552.0 * std::sqrt(4186.0)) * sh_11_12 * z
    + Element(5.0 / 138.0 * std::sqrt(805.0)) * sh_11_13 * y
    - Element(5.0 / 184.0 * std::sqrt(230.0)) * sh_11_14 * z
    - Element(5.0 / 184.0 * std::sqrt(230.0)) * sh_11_8 * x;
const Element sh_12_15 = Element(5.0 / 552.0 * std::sqrt(4830.0)) * sh_11_13 * z
    + Element(5.0 / 92.0 * std::sqrt(345.0)) * sh_11_14 * y
    - Element(5.0 / 92.0 * std::sqrt(46.0)) * sh_11_15 * z
    - Element(5.0 / 92.0 * std::sqrt(46.0)) * sh_11_7 * x
    - Element(5.0 / 552.0 * std::sqrt(4830.0)) * sh_11_9 * x;
const Element sh_12_16 = Element(5.0 / 138.0 * std::sqrt(345.0)) * sh_11_14 * z
    + Element(10.0 / 69.0 * std::sqrt(46.0)) * sh_11_15 * y
    - Element(5.0 / 276.0 * std::sqrt(322.0)) * sh_11_16 * z
    - Element(5.0 / 276.0 * std::sqrt(322.0)) * sh_11_6 * x
    - Element(5.0 / 138.0 * std::sqrt(345.0)) * sh_11_8 * x;
const Element sh_12_17 = Element(5.0 / 138.0 * std::sqrt(391.0)) * sh_11_15 * z
    + Element(5.0 / 276.0 * std::sqrt(2737.0)) * sh_11_16 * y
    - Element(5.0 / 552.0 * std::sqrt(966.0)) * sh_11_17 * z
    - Element(5.0 / 552.0 * std::sqrt(966.0)) * sh_11_5 * x
    - Element(5.0 / 138.0 * std::sqrt(391.0)) * sh_11_7 * x;
const Element sh_12_18 = Element(5.0 / 552.0 * std::sqrt(3.0)) * (Element(std::sqrt(2346.0)) * sh_11_16 * z + Element(12.0 * std::sqrt(23.0)) * sh_11_17 * y - Element(std::sqrt(230.0)) * sh_11_18 * z - Element(std::sqrt(230.0)) * sh_11_4 * x - Element(std::sqrt(2346.0)) * sh_11_6 * x);
const Element sh_12_19 = Element(5.0 / 184.0 * std::sqrt(874.0)) * sh_11_17 * z
    + Element(5.0 / 276.0 * std::sqrt(2185.0)) * sh_11_18 * y
    - Element(5.0 / 276.0 * std::sqrt(115.0)) * sh_11_19 * z
    - Element(5.0 / 276.0 * std::sqrt(115.0)) * sh_11_3 * x
    - Element(5.0 / 184.0 * std::sqrt(874.0)) * sh_11_5 * x;
const Element sh_12_20 = Element(5.0 / 276.0 * std::sqrt(2185.0)) * sh_11_18 * z
    + Element(5.0 / 69.0 * std::sqrt(115.0)) * sh_11_19 * y
    - Element(5.0 / 276.0 * std::sqrt(69.0)) * sh_11_2 * x
    - Element(5.0 / 276.0 * std::sqrt(69.0)) * sh_11_20 * z
    - Element(5.0 / 276.0 * std::sqrt(2185.0)) * sh_11_4 * x;
const Element sh_12_21 = Element(-5.0 / 552.0 * std::sqrt(138.0)) * sh_11_1 * x
    + Element(5.0 / 276.0 * std::sqrt(2415.0)) * sh_11_19 * z
    + Element(5.0 / 92.0 * std::sqrt(161.0)) * sh_11_20 * y
    - Element(5.0 / 552.0 * std::sqrt(138.0)) * sh_11_21 * z
    - Element(5.0 / 276.0 * std::sqrt(2415.0)) * sh_11_3 * x;
const Element sh_12_22 = Element(-5.0 / 552.0 * std::sqrt(46.0)) * sh_11_0 * x
    - Element(5.0 / 552.0 * std::sqrt(10626.0)) * sh_11_2 * x
    + Element(5.0 / 552.0 * std::sqrt(10626.0)) * sh_11_20 * z
    + Element(5.0 / 138.0 * std::sqrt(253.0)) * sh_11_21 * y
    - Element(5.0 / 552.0 * std::sqrt(46.0)) * sh_11_22 * z;
const Element sh_12_23 = Element(-5.0 / 24.0 * std::sqrt(22.0)) * sh_11_1 * x
    + Element(5.0 / 24.0 * std::sqrt(22.0)) * sh_11_21 * z
    + Element(5.0 / 12.0) * sh_11_22 * y;
const Element sh_12_24 = Element(5.0 / 12.0 * std::sqrt(6.0)) * (-sh_11_0 * x + sh_11_22 * z);
"""

NORMALIZE_COORDINATES = r"""
const Element length = vector_norm(x, y, z);
const Element denominator = clamp_min(length, Element(1e-12));
const Element nx = x / denominator;
const Element ny = y / denominator;
const Element nz = z / denominator;
"""
SCALAR_KERNEL = r"""{layout_declaration}__global__ __launch_bounds__(128,1) void {kernel}({parameters},int64_t num_items{layout_parameter}) {{
    for(Index item=Index(blockIdx.x)*128+threadIdx.x;item<num_items;item+=Index(gridDim.x)*128) {{
{item_setup}        {expressions}
    }}
}}
"""
SOURCE_HEADER = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
using Element={scalar};
"""


class TorchExpression(ast.NodeTransformer):
    """Keep CUDA and live Torch replay on the same ordered polynomial."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == 'Element':
                return node.args[0]
            if node.func.id == 'pow':
                return ast.Call(
                    func=ast.Attribute(value=node.args[0], attr='pow', ctx=ast.Load()),
                    args=node.args[1:], keywords=[],
                )
        return node


@lru_cache(maxsize=1)
def polynomial_expressions():
    expressions = []
    degree = 0
    for name, expression in cuda_assignments(CARTESIAN_POLYNOMIAL):
        if name.startswith('sh_'):
            degree = int(name.split('_')[1])
        expressions.append((degree, name, expression))
    return tuple(expressions)


@lru_cache(maxsize=1)
def compiled_expressions():
    expressions = []
    for degree, name, expression in polynomial_expressions():
        # Degree zero must remain a Tensor for CatBackward/SelectBackward replay.
        if name == 'sh_0_0':
            expression = ast.parse('torch.ones_like(x)', mode='eval').body
        expression = TorchExpression().visit(ast.parse(ast.unparse(expression), mode='eval'))
        expression = ast.fix_missing_locations(expression)
        expressions.append((degree, name, compile(expression, '<cartesian-cuda>', 'eval')))
    return tuple(expressions)


def evaluate_polynomial(coordinates, degrees, normalization, *, evaluator=eval):
    """Evaluate the bundled expressions with their original operation order.

    Metadata tracing uses a restricted namespace. Live tensor replay
    explicitly supplies its normal Python evaluator for live Torch tensors.
    """
    variables = {name: coordinates[..., column] for column, name in enumerate(('x', 'y', 'z'))}
    for degree, name, expression in compiled_expressions():
        if degree > max(degrees):
            break
        variables[name] = evaluator(
            expression, {'__builtins__': {}, 'math': math, 'torch': torch}, variables
        )
    # Stack the complete range first.  Omitting unrequested columns here would
    # change SelectBackward/CatBackward and the order of gradient additions.
    result = torch.stack(
        [
            variables[f'sh_{degree}_{column}']
            for degree in range(max(degrees) + 1)
            for column in range(2 * degree + 1)
        ],
        -1,
    )
    if tuple(degrees) != tuple(range(max(degrees) + 1)):
        result = torch.cat(
            [result[..., degree * degree : (degree + 1) * (degree + 1)] for degree in degrees], -1
        )
    if normalization == 'integral':
        result.div_(math.sqrt(4 * math.pi))
    elif normalization == 'norm':
        result.div_(
            torch.cat(
                [
                    math.sqrt(2 * degree + 1)
                    * torch.ones(2 * degree + 1, dtype=result.dtype, device=result.device)
                    for degree in degrees
                ]
            )
        )
    return result


def build_forward_graph(meta):
    """Build the ordered Cartesian polynomial and normalization graph."""
    graph = ScalarGraph()
    # Reserve all tensor roles in stable order for native source/cache identity.
    x = tuple(graph.input('x', i) for i in range(3))
    grad_out = tuple(graph.input('grad_out', i) for i in range(meta['out_dim']))
    rx = tuple(graph.input('rx', i) for i in range(3))
    bx = tuple(graph.input('bx', i) for i in range(3))
    bg = tuple(graph.input('bg', i) for i in range(meta['out_dim']))
    coordinates = x
    if meta['normalize'] and max(meta['degrees']) > 0:
        normalized = dict(zip(('x', 'y', 'z'), x))
        for name, expression in cuda_assignments(NORMALIZE_COORDINATES):
            normalized[name] = lower_expression(graph, expression, normalized)
        coordinates = tuple(normalized[name] for name in ('nx', 'ny', 'nz'))
    variables = dict(zip(('x', 'y', 'z'), coordinates))
    for degree, name, expression in polynomial_expressions():
        if degree > max(meta['degrees']):
            break
        variables[name] = lower_expression(graph, expression, variables)
    output = []
    for degree in meta['degrees']:
        divisor = (
            math.sqrt(4 * math.pi)
            if meta['normalization'] == 'integral'
            else math.sqrt(2 * degree + 1) if meta['normalization'] == 'norm' else None
        )
        for component in range(2 * degree + 1):
            value = variables[f'sh_{degree}_{component}']
            if meta['normalization'] == 'integral':
                # ATen's Python-scalar division computes the reciprocal in
                # double before casting once to the operation's dtype.
                value = graph.binary('mul', value, graph.constant(1.0 / divisor))
            elif divisor is not None:
                # Norm scaling uses a tensor denominator in the reference.
                value = graph.binary('div', value, graph.constant(divisor))
            output.append(value)
    return graph, {'out': tuple(output)}


def build_forward_program(meta, *, needs=None, available=None):
    available = {'x'} if available is None else set(available)
    if available.difference({'x'}):
        raise ValueError('Unknown spherical-harmonic forward inputs')
    active = 'x' in available
    return dict(
        order=0,
        outputs=('out',),
        active_outputs=('out',) if active else (),
        zero_outputs=(),
        inputs=('x',) if active else (),
        terms={},
        live_terms=(),
    )


def make_scalar_source(
    meta,
    program,
    *,
    graph=None,
    outputs=None,
    input_addresses=None,
    layout_declaration='',
    layout_parameter='',
    item_setup='',
    layout_setup='',
    layout_argument='',
):
    """Emit the ordered scalar CUDA program and its native launch.

    Backward owns derivative graph construction and uses this same emitter.
    The default graph is the forward polynomial.
    """
    if graph is None:
        graph, outputs = build_forward_graph(meta)
    names = tuple(program['active_outputs'])
    if not names or not program['inputs']:
        raise ValueError('An active native spherical-harmonic program needs a tensor dependency')
    selected = tuple(n for name in names for n in outputs[name])
    ancestors = graph.ancestors(selected)
    nodes = [n for n in graph.nodes if n.index in ancestors]
    dependencies = {n.value[0] for n in nodes if n.op == 'input'}
    if not dependencies.issubset(program['inputs']):
        raise ValueError(
            f'Program omitted live dependencies: {dependencies.difference(program["inputs"])}'
        )
    is_double = meta['dtype'] == torch.float64
    scalar = 'double' if is_double else 'float'
    arithmetic = {
        'add': '__dadd_rn' if is_double else '__fadd_rn',
        'sub': '__dsub_rn' if is_double else '__fsub_rn',
        'mul': '__dmul_rn' if is_double else '__fmul_rn',
        'div': '__ddiv_rn' if is_double else '__fdiv_rn',
    }


    lines = []
    for n in nodes:
        if n.op == 'constant':
            continue
        a = tuple(
            float_literal(float.fromhex(x.value), meta['dtype']) if x.op == 'constant' else f'v{x.index}'
            for x in n.args
        )
        type_ = 'bool' if n.op in ('eq', 'ge') else 'Element'
        if n.op == 'input':
            role, column = n.value
            dimension = meta['out_dim'] if role in ('grad_out', 'bg') else 3
            value = f'{role}[item*{dimension}+{column}]'
            if input_addresses is not None:
                value = input_addresses.get((role, column), value)
        elif n.op in arithmetic:
            value = f'{arithmetic[n.op]}({a[0]},{a[1]})'
        elif n.op == 'neg':
            value = f'(-{a[0]})'
        elif n.op == 'pow':
            value = a[0]
            for _ in range(n.value - 1):
                value = f"{arithmetic['mul']}({value},{a[0]})"
        elif n.op == 'norm':
            # ATen's three-coordinate reduction splits X/Z across one lane
            # when the reduced axis strides fastest. A faster batch axis
            # keeps all three coordinates in one thread in X/Y/Z order.
            layout = n.value if n.value is not None else meta.get('norm_layout', 'xz_y')
            second, third = (2, 1) if layout == 'xz_y' else (1, 2)
            sqrt = '__dsqrt_rn' if is_double else '__fsqrt_rn'
            add, mul = arithmetic['add'], arithmetic['mul']
            value = (
                f'{sqrt}({add}({add}({mul}({a[0]},{a[0]}),'
                f'{mul}({a[second]},{a[second]})),{mul}({a[third]},{a[third]})))'
            )
        elif n.op == 'clamp':
            value = f'({a[0]}<{a[1]}?{a[1]}:{a[0]})'
        elif n.op == 'where':
            value = f'({a[0]}?{a[1]}:{a[2]})'
        elif n.op == 'eq':
            value = f'({a[0]}=={a[1]})'
        elif n.op == 'ge':
            value = f'({a[0]}>={a[1]})'
        else:
            raise ValueError(n.op)
        lines.append(f'const {type_} v{n.index}={value};')
    for name in names:
        values = tuple(
            float_literal(float.fromhex(value.value), meta['dtype'])
            if value.op == 'constant' else f'v{value.index}'
            for value in outputs[name]
        )
        width = 2 if is_double else 4
        if program['order'] == 0 and len(values) % width == 0:
            vector = 'double2' if is_double else 'float4'
            lines.extend(
                f'reinterpret_cast<{vector}*>(out_out)[item*{len(values)//width}+{i//width}]='
                f'make_{vector}({",".join(values[i:i+width])});'
                for i in range(0, len(values), width)
            )
        else:
            lines.extend(
                f'out_{name}[item*{len(values)}+{i}]={value};'
                for i, value in enumerate(values)
            )
    parameters = [f'const Element* __restrict__ {role}' for role in program['inputs']]
    parameters += [f'Element* __restrict__ out_{name}' for name in names]
    arguments = [f'{role}.data_ptr<Element>()' for role in program['inputs']]
    arguments += [f'result_{name}.data_ptr<Element>()' for name in names]
    allocations = [
        f'auto result_{name}=torch::empty({{num_items,int64_t({len(outputs[name])})}},options);'
        for name in names
    ]
    returns = [
        f'pybind11::cast(result_{name})' if name in names else 'pybind11::none()'
        for name in program['outputs']
    ]
    phase = ('forward', 'backward', 'double_backward', 'triple_backward')[program['order']]
    kernel = f'spherical_harmonics_{phase}'
    first = program['inputs'][0]
    return (SOURCE_HEADER.format(scalar=scalar)
        + SCALAR_KERNEL.format(
            layout_declaration=layout_declaration, kernel=kernel,
            parameters=','.join(parameters), layout_parameter=layout_parameter,
            item_setup=item_setup, expressions="\n".join(lines),
        )
        + HOST_BINDING.format(
            launch_parameters=','.join(f'torch::Tensor {role}' for role in program['inputs']),
            first=first, allocations="\n".join(allocations),
            layout_setup=layout_setup, kernel=kernel, arguments=','.join(arguments),
            layout_argument=layout_argument, results=','.join(returns),
        ))


def make_strided_source(meta, program, batch_rank):
    if program["order"] != 0 or program["inputs"] != ("x",) or not 0 <= batch_rank <= 63:
        raise ValueError("Strided SH supports only the complete forward X input")
    fields = "\n".join(f"    int64_t size_{i},stride_{i};" for i in range(batch_rank))
    descriptor = "struct InputLayout {\n" + fields + "\n    int64_t coordinate_stride;\n};\n"
    offset = "        int64_t input_offset=0,residual=item;\n"
    for i in range(batch_rank - 1, 0, -1):
        offset += f"        input_offset+=(residual%layout.size_{i})*layout.stride_{i}; residual/=layout.size_{i};\n"
    if batch_rank:
        offset += "        input_offset+=residual*layout.stride_0;\n"
    addresses = {
        ('x', column): f"x[input_offset+int64_t({column})*layout.coordinate_stride]"
        for column in range(3)
    }
    init = "    InputLayout layout;\n"
    for i in range(batch_rank):
        init += f"    layout.size_{i}=x.size({i}); layout.stride_{i}=x.stride({i});\n"
    init += f"    layout.coordinate_stride=x.stride({batch_rank});\n"
    return make_scalar_source(
        meta,
        program,
        input_addresses=addresses,
        layout_declaration=descriptor,
        layout_parameter=',InputLayout layout',
        item_setup=offset,
        layout_setup=init,
        layout_argument=',layout',
    )


def normalization_layout(x):
    batch_strides = [stride for size, stride in zip(x.shape[:-1], x.stride()[:-1]) if size > 1]
    return 'xz_y' if not batch_strides or x.stride(-1) < batch_strides[-1] else 'xy_z'


def run_forward(op, x, norm_layout=None):
    """Launch contiguous or strided Cartesian forward without saving intermediates."""
    batch = x.shape[:-1]
    num_items = prod(batch)
    if num_items == 0:
        return x.new_empty(batch + (op.meta['out_dim'],))

    direct_input = x.ndim <= 64 and not x.is_contiguous()
    input_span = 0
    if direct_input and x.numel():
        input_span = 1 + sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
    bound = max(max(3, op.meta['out_dim']) * num_items, input_span) + 65535 * 128
    if bound > (1 << 63) - 1:
        raise OverflowError('SphericalHarmonics offsets exceed signed int64 capacity')
    index_bits = 64 if bound > (1 << 31) - 1 else op.index_bits
    arch = get_cuda_arch(x.device)
    if op.normalize and op._lmax > 0:
        norm_layout = normalization_layout(x) if norm_layout is None else norm_layout
    else:
        norm_layout = None
    meta = op.meta

    program_key = ('program', 0, None, ('x',))
    record = op.kernels.get(program_key)
    if record is None:
        if norm_layout is not None:
            meta = {**op.meta, 'norm_layout': norm_layout}
        program = build_forward_program(meta)
        record = {'program': program, 'save_roles': program['inputs'], 'launch': None}
        op.kernels[program_key] = record
        while len(op.kernels) > 128:
            op.kernels.popitem(last=False)
    else:
        op.kernels.move_to_end(program_key)

    program = record['program']
    backward_layout = (tuple(x.shape), tuple(x.stride())) if direct_input and norm_layout else None
    key = (0, None, ('x',), index_bits, arch, norm_layout, backward_layout, x.ndim if direct_input else None)
    launch_record = op.kernels.get(key)
    if launch_record is None:
        if norm_layout is not None:
            meta = {**op.meta, 'norm_layout': norm_layout}
        if direct_input:
            prefix = 'spherical_harmonics_strided_forward'
            source = make_strided_source(meta, program, x.ndim - 1)
        else:
            precision = 'fp64' if meta['dtype'] == torch.float64 else 'fp32'
            prefix = f'spherical_harmonics_forward_{precision}'
            source = make_scalar_source(meta, program)
        if op._lmax > 0:
            from .backward import make_backward_source, program_record

            backward_record = program_record(op, 1)
            backward_program = backward_record['program']
            layout_specs, layout_key = None, None
            if norm_layout is not None:
                if x.is_contiguous():
                    layout_specs = {
                        'x': dict(shape=(2, 3), stride=(3, 1)),
                        'grad_out': dict(shape=(2, meta['out_dim']), stride=(meta['out_dim'], 1)),
                    }
                    layout_key = 'contiguous'
                else:
                    grad_shape = (*batch, meta['out_dim'])
                    grad_strides, stride = [], 1
                    for size in reversed(grad_shape):
                        grad_strides.append(stride)
                        stride *= max(1, size)
                    layout_specs = {
                        'x': dict(shape=tuple(x.shape), stride=tuple(x.stride())),
                        'grad_out': dict(shape=grad_shape, stride=tuple(reversed(grad_strides))),
                    }
                    layout_key = tuple(
                        (name, spec['shape'], spec['stride']) for name, spec in layout_specs.items()
                    )
            source = combine_cuda_sources({
                'forward': source,
                'backward': make_backward_source(meta, backward_program, layout_specs=layout_specs),
            })
            extension = load_cuda_extension(
                prefix.replace('forward', 'forward_backward'), source,
                index_bits=index_bits, arch=arch, use_fast_math=False, minimal_headers=True,
            )
            launch = extension.forward.launch
            backward_bound = max(3, meta['out_dim']) * num_items + 65535 * 128
            backward_bits = 64 if backward_bound > (1 << 31) - 1 else op.index_bits
            if backward_bits != index_bits:
                extension = load_cuda_extension(
                    prefix.replace('forward', 'forward_backward'), source,
                    index_bits=backward_bits, arch=arch, use_fast_math=False, minimal_headers=True,
                )
            # Degree-one polynomials do not save X for their constant derivative.
            inputs = [('x', 'grad_out')]
            if 'x' not in backward_program['inputs']:
                inputs.append(('grad_out',))
            for available in inputs:
                backward_key = (1, None, available, backward_bits, arch, norm_layout, layout_key, None)
                op.kernels[backward_key] = {**backward_record, 'launch': extension.backward.launch}
        else:
            launch = load_cuda_extension(
                prefix, source, index_bits=index_bits, arch=arch,
                use_fast_math=False, minimal_headers=True,
            ).launch
        op.kernels[key] = {**record, 'launch': launch}
        while len(op.kernels) > 128:
            op.kernels.popitem(last=False)
    else:
        launch = launch_record['launch']
        op.kernels.move_to_end(key)

    (result,) = launch(x if direct_input else x.contiguous(), num_items)
    return result.reshape(batch + (result.shape[-1],))
