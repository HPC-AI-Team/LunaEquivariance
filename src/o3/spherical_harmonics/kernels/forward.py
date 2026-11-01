"""Cartesian spherical-harmonic CUDA, degrees 0 through 12."""

from __future__ import annotations

import ast
from functools import lru_cache
import math
from textwrap import indent
from math import prod

import torch

from ....utils.extension import get_cuda_arch, combine_cuda_sources, load_cuda_extension
from ....utils.precision import float_literal
from .scalar import ScalarGraph, cuda_assignments, lower_expression
from .utils import (
    BLOCK_THREADS, OUTPUT_TILE_COLUMNS, TRIPLE_BACKWARD_MIN_BLOCKS, HOST_BINDING,
)


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
const Element sh_3_0 = Element(std::sqrt(7.0 / 6.0)) * (sh_2_0 * z + sh_2_4 * x);
const Element radius2 = x2z2 + y2;
const Element sh_3_1 = Element(std::sqrt(7.0)) * y * sh_2_0;
const Element sh_3_2 = Element(std::sqrt(35.0 / 8.0)) * y * sh_2_1 - Element(std::sqrt(21.0 / 24.0)) * radius2 * sh_1_0;
const Element sh_3_3 = Element(std::sqrt(35.0 / 9.0)) * y * sh_2_2 - Element(std::sqrt(28.0 / 27.0)) * radius2 * sh_1_1;
const Element sh_3_4 = Element(std::sqrt(35.0 / 8.0)) * y * sh_2_3 - Element(std::sqrt(21.0 / 24.0)) * radius2 * sh_1_2;
const Element sh_3_5 = Element(std::sqrt(7.0)) * y * sh_2_4;
const Element sh_3_6 = Element(std::sqrt(7.0 / 6.0)) * (sh_2_4 * z - sh_2_0 * x);

// Degree 4
const Element sh_4_0 = Element(std::sqrt(9.0 / 8.0)) * (sh_3_0 * z + sh_3_6 * x);
const Element sh_4_1 = Element(std::sqrt(9.0)) * y * sh_3_0;
const Element sh_4_2 = Element(std::sqrt(63.0 / 12.0)) * y * sh_3_1 - Element(std::sqrt(45.0 / 60.0)) * radius2 * sh_2_0;
const Element sh_4_3 = Element(std::sqrt(63.0 / 15.0)) * y * sh_3_2 - Element(std::sqrt(72.0 / 75.0)) * radius2 * sh_2_1;
const Element sh_4_4 = Element(std::sqrt(63.0 / 16.0)) * y * sh_3_3 - Element(std::sqrt(81.0 / 80.0)) * radius2 * sh_2_2;
const Element sh_4_5 = Element(std::sqrt(63.0 / 15.0)) * y * sh_3_4 - Element(std::sqrt(72.0 / 75.0)) * radius2 * sh_2_3;
const Element sh_4_6 = Element(std::sqrt(63.0 / 12.0)) * y * sh_3_5 - Element(std::sqrt(45.0 / 60.0)) * radius2 * sh_2_4;
const Element sh_4_7 = Element(std::sqrt(9.0)) * y * sh_3_6;
const Element sh_4_8 = Element(std::sqrt(9.0 / 8.0)) * (sh_3_6 * z - sh_3_0 * x);

// Degree 5
const Element sh_5_0 = Element(std::sqrt(11.0 / 10.0)) * (sh_4_0 * z + sh_4_8 * x);
const Element sh_5_1 = Element(std::sqrt(11.0)) * y * sh_4_0;
const Element sh_5_2 = Element(std::sqrt(99.0 / 16.0)) * y * sh_4_1 - Element(std::sqrt(77.0 / 112.0)) * radius2 * sh_3_0;
const Element sh_5_3 = Element(std::sqrt(99.0 / 21.0)) * y * sh_4_2 - Element(std::sqrt(132.0 / 147.0)) * radius2 * sh_3_1;
const Element sh_5_4 = Element(std::sqrt(99.0 / 24.0)) * y * sh_4_3 - Element(std::sqrt(165.0 / 168.0)) * radius2 * sh_3_2;
const Element sh_5_5 = Element(std::sqrt(99.0 / 25.0)) * y * sh_4_4 - Element(std::sqrt(176.0 / 175.0)) * radius2 * sh_3_3;
const Element sh_5_6 = Element(std::sqrt(99.0 / 24.0)) * y * sh_4_5 - Element(std::sqrt(165.0 / 168.0)) * radius2 * sh_3_4;
const Element sh_5_7 = Element(std::sqrt(99.0 / 21.0)) * y * sh_4_6 - Element(std::sqrt(132.0 / 147.0)) * radius2 * sh_3_5;
const Element sh_5_8 = Element(std::sqrt(99.0 / 16.0)) * y * sh_4_7 - Element(std::sqrt(77.0 / 112.0)) * radius2 * sh_3_6;
const Element sh_5_9 = Element(std::sqrt(11.0)) * y * sh_4_8;
const Element sh_5_10 = Element(std::sqrt(11.0 / 10.0)) * (sh_4_8 * z - sh_4_0 * x);

// Degree 6
const Element sh_6_0 = Element(std::sqrt(13.0 / 12.0)) * (sh_5_0 * z + sh_5_10 * x);
const Element sh_6_1 = Element(std::sqrt(13.0)) * y * sh_5_0;
const Element sh_6_2 = Element(std::sqrt(143.0 / 20.0)) * y * sh_5_1 - Element(std::sqrt(117.0 / 180.0)) * radius2 * sh_4_0;
const Element sh_6_3 = Element(std::sqrt(143.0 / 27.0)) * y * sh_5_2 - Element(std::sqrt(208.0 / 243.0)) * radius2 * sh_4_1;
const Element sh_6_4 = Element(std::sqrt(143.0 / 32.0)) * y * sh_5_3 - Element(std::sqrt(273.0 / 288.0)) * radius2 * sh_4_2;
const Element sh_6_5 = Element(std::sqrt(143.0 / 35.0)) * y * sh_5_4 - Element(std::sqrt(312.0 / 315.0)) * radius2 * sh_4_3;
const Element sh_6_6 = Element(std::sqrt(143.0 / 36.0)) * y * sh_5_5 - Element(std::sqrt(325.0 / 324.0)) * radius2 * sh_4_4;
const Element sh_6_7 = Element(std::sqrt(143.0 / 35.0)) * y * sh_5_6 - Element(std::sqrt(312.0 / 315.0)) * radius2 * sh_4_5;
const Element sh_6_8 = Element(std::sqrt(143.0 / 32.0)) * y * sh_5_7 - Element(std::sqrt(273.0 / 288.0)) * radius2 * sh_4_6;
const Element sh_6_9 = Element(std::sqrt(143.0 / 27.0)) * y * sh_5_8 - Element(std::sqrt(208.0 / 243.0)) * radius2 * sh_4_7;
const Element sh_6_10 = Element(std::sqrt(143.0 / 20.0)) * y * sh_5_9 - Element(std::sqrt(117.0 / 180.0)) * radius2 * sh_4_8;
const Element sh_6_11 = Element(std::sqrt(13.0)) * y * sh_5_10;
const Element sh_6_12 = Element(std::sqrt(13.0 / 12.0)) * (sh_5_10 * z - sh_5_0 * x);

// Degree 7
const Element sh_7_0 = Element(std::sqrt(15.0 / 14.0)) * (sh_6_0 * z + sh_6_12 * x);
const Element sh_7_1 = Element(std::sqrt(15.0)) * y * sh_6_0;
const Element sh_7_2 = Element(std::sqrt(195.0 / 24.0)) * y * sh_6_1 - Element(std::sqrt(165.0 / 264.0)) * radius2 * sh_5_0;
const Element sh_7_3 = Element(std::sqrt(195.0 / 33.0)) * y * sh_6_2 - Element(std::sqrt(300.0 / 363.0)) * radius2 * sh_5_1;
const Element sh_7_4 = Element(std::sqrt(195.0 / 40.0)) * y * sh_6_3 - Element(std::sqrt(405.0 / 440.0)) * radius2 * sh_5_2;
const Element sh_7_5 = Element(std::sqrt(195.0 / 45.0)) * y * sh_6_4 - Element(std::sqrt(480.0 / 495.0)) * radius2 * sh_5_3;
const Element sh_7_6 = Element(std::sqrt(195.0 / 48.0)) * y * sh_6_5 - Element(std::sqrt(525.0 / 528.0)) * radius2 * sh_5_4;
const Element sh_7_7 = Element(std::sqrt(195.0 / 49.0)) * y * sh_6_6 - Element(std::sqrt(540.0 / 539.0)) * radius2 * sh_5_5;
const Element sh_7_8 = Element(std::sqrt(195.0 / 48.0)) * y * sh_6_7 - Element(std::sqrt(525.0 / 528.0)) * radius2 * sh_5_6;
const Element sh_7_9 = Element(std::sqrt(195.0 / 45.0)) * y * sh_6_8 - Element(std::sqrt(480.0 / 495.0)) * radius2 * sh_5_7;
const Element sh_7_10 = Element(std::sqrt(195.0 / 40.0)) * y * sh_6_9 - Element(std::sqrt(405.0 / 440.0)) * radius2 * sh_5_8;
const Element sh_7_11 = Element(std::sqrt(195.0 / 33.0)) * y * sh_6_10 - Element(std::sqrt(300.0 / 363.0)) * radius2 * sh_5_9;
const Element sh_7_12 = Element(std::sqrt(195.0 / 24.0)) * y * sh_6_11 - Element(std::sqrt(165.0 / 264.0)) * radius2 * sh_5_10;
const Element sh_7_13 = Element(std::sqrt(15.0)) * y * sh_6_12;
const Element sh_7_14 = Element(std::sqrt(15.0 / 14.0)) * (sh_6_12 * z - sh_6_0 * x);

// Degree 8
const Element sh_8_0 = Element(std::sqrt(17.0 / 16.0)) * (sh_7_0 * z + sh_7_14 * x);
const Element sh_8_1 = Element(std::sqrt(17.0)) * y * sh_7_0;
const Element sh_8_2 = Element(std::sqrt(255.0 / 28.0)) * y * sh_7_1 - Element(std::sqrt(221.0 / 364.0)) * radius2 * sh_6_0;
const Element sh_8_3 = Element(std::sqrt(255.0 / 39.0)) * y * sh_7_2 - Element(std::sqrt(408.0 / 507.0)) * radius2 * sh_6_1;
const Element sh_8_4 = Element(std::sqrt(255.0 / 48.0)) * y * sh_7_3 - Element(std::sqrt(561.0 / 624.0)) * radius2 * sh_6_2;
const Element sh_8_5 = Element(std::sqrt(255.0 / 55.0)) * y * sh_7_4 - Element(std::sqrt(680.0 / 715.0)) * radius2 * sh_6_3;
const Element sh_8_6 = Element(std::sqrt(255.0 / 60.0)) * y * sh_7_5 - Element(std::sqrt(765.0 / 780.0)) * radius2 * sh_6_4;
const Element sh_8_7 = Element(std::sqrt(255.0 / 63.0)) * y * sh_7_6 - Element(std::sqrt(816.0 / 819.0)) * radius2 * sh_6_5;
const Element sh_8_8 = Element(std::sqrt(255.0 / 64.0)) * y * sh_7_7 - Element(std::sqrt(833.0 / 832.0)) * radius2 * sh_6_6;
const Element sh_8_9 = Element(std::sqrt(255.0 / 63.0)) * y * sh_7_8 - Element(std::sqrt(816.0 / 819.0)) * radius2 * sh_6_7;
const Element sh_8_10 = Element(std::sqrt(255.0 / 60.0)) * y * sh_7_9 - Element(std::sqrt(765.0 / 780.0)) * radius2 * sh_6_8;
const Element sh_8_11 = Element(std::sqrt(255.0 / 55.0)) * y * sh_7_10 - Element(std::sqrt(680.0 / 715.0)) * radius2 * sh_6_9;
const Element sh_8_12 = Element(std::sqrt(255.0 / 48.0)) * y * sh_7_11 - Element(std::sqrt(561.0 / 624.0)) * radius2 * sh_6_10;
const Element sh_8_13 = Element(std::sqrt(255.0 / 39.0)) * y * sh_7_12 - Element(std::sqrt(408.0 / 507.0)) * radius2 * sh_6_11;
const Element sh_8_14 = Element(std::sqrt(255.0 / 28.0)) * y * sh_7_13 - Element(std::sqrt(221.0 / 364.0)) * radius2 * sh_6_12;
const Element sh_8_15 = Element(std::sqrt(17.0)) * y * sh_7_14;
const Element sh_8_16 = Element(std::sqrt(17.0 / 16.0)) * (sh_7_14 * z - sh_7_0 * x);

// Degree 9
const Element sh_9_0 = Element(std::sqrt(19.0 / 18.0)) * (sh_8_0 * z + sh_8_16 * x);
const Element sh_9_1 = Element(std::sqrt(19.0)) * y * sh_8_0;
const Element sh_9_2 = Element(std::sqrt(323.0 / 32.0)) * y * sh_8_1 - Element(std::sqrt(285.0 / 480.0)) * radius2 * sh_7_0;
const Element sh_9_3 = Element(std::sqrt(323.0 / 45.0)) * y * sh_8_2 - Element(std::sqrt(532.0 / 675.0)) * radius2 * sh_7_1;
const Element sh_9_4 = Element(std::sqrt(323.0 / 56.0)) * y * sh_8_3 - Element(std::sqrt(741.0 / 840.0)) * radius2 * sh_7_2;
const Element sh_9_5 = Element(std::sqrt(323.0 / 65.0)) * y * sh_8_4 - Element(std::sqrt(912.0 / 975.0)) * radius2 * sh_7_3;
const Element sh_9_6 = Element(std::sqrt(323.0 / 72.0)) * y * sh_8_5 - Element(std::sqrt(1045.0 / 1080.0)) * radius2 * sh_7_4;
const Element sh_9_7 = Element(std::sqrt(323.0 / 77.0)) * y * sh_8_6 - Element(std::sqrt(1140.0 / 1155.0)) * radius2 * sh_7_5;
const Element sh_9_8 = Element(std::sqrt(323.0 / 80.0)) * y * sh_8_7 - Element(std::sqrt(1197.0 / 1200.0)) * radius2 * sh_7_6;
const Element sh_9_9 = Element(std::sqrt(323.0 / 81.0)) * y * sh_8_8 - Element(std::sqrt(1216.0 / 1215.0)) * radius2 * sh_7_7;
const Element sh_9_10 = Element(std::sqrt(323.0 / 80.0)) * y * sh_8_9 - Element(std::sqrt(1197.0 / 1200.0)) * radius2 * sh_7_8;
const Element sh_9_11 = Element(std::sqrt(323.0 / 77.0)) * y * sh_8_10 - Element(std::sqrt(1140.0 / 1155.0)) * radius2 * sh_7_9;
const Element sh_9_12 = Element(std::sqrt(323.0 / 72.0)) * y * sh_8_11 - Element(std::sqrt(1045.0 / 1080.0)) * radius2 * sh_7_10;
const Element sh_9_13 = Element(std::sqrt(323.0 / 65.0)) * y * sh_8_12 - Element(std::sqrt(912.0 / 975.0)) * radius2 * sh_7_11;
const Element sh_9_14 = Element(std::sqrt(323.0 / 56.0)) * y * sh_8_13 - Element(std::sqrt(741.0 / 840.0)) * radius2 * sh_7_12;
const Element sh_9_15 = Element(std::sqrt(323.0 / 45.0)) * y * sh_8_14 - Element(std::sqrt(532.0 / 675.0)) * radius2 * sh_7_13;
const Element sh_9_16 = Element(std::sqrt(323.0 / 32.0)) * y * sh_8_15 - Element(std::sqrt(285.0 / 480.0)) * radius2 * sh_7_14;
const Element sh_9_17 = Element(std::sqrt(19.0)) * y * sh_8_16;
const Element sh_9_18 = Element(std::sqrt(19.0 / 18.0)) * (sh_8_16 * z - sh_8_0 * x);

// Degree 10
const Element sh_10_0 = Element(std::sqrt(21.0 / 20.0)) * (sh_9_0 * z + sh_9_18 * x);
const Element sh_10_1 = Element(std::sqrt(21.0)) * y * sh_9_0;
const Element sh_10_2 = Element(std::sqrt(399.0 / 36.0)) * y * sh_9_1 - Element(std::sqrt(357.0 / 612.0)) * radius2 * sh_8_0;
const Element sh_10_3 = Element(std::sqrt(399.0 / 51.0)) * y * sh_9_2 - Element(std::sqrt(672.0 / 867.0)) * radius2 * sh_8_1;
const Element sh_10_4 = Element(std::sqrt(399.0 / 64.0)) * y * sh_9_3 - Element(std::sqrt(945.0 / 1088.0)) * radius2 * sh_8_2;
const Element sh_10_5 = Element(std::sqrt(399.0 / 75.0)) * y * sh_9_4 - Element(std::sqrt(1176.0 / 1275.0)) * radius2 * sh_8_3;
const Element sh_10_6 = Element(std::sqrt(399.0 / 84.0)) * y * sh_9_5 - Element(std::sqrt(1365.0 / 1428.0)) * radius2 * sh_8_4;
const Element sh_10_7 = Element(std::sqrt(399.0 / 91.0)) * y * sh_9_6 - Element(std::sqrt(1512.0 / 1547.0)) * radius2 * sh_8_5;
const Element sh_10_8 = Element(std::sqrt(399.0 / 96.0)) * y * sh_9_7 - Element(std::sqrt(1617.0 / 1632.0)) * radius2 * sh_8_6;
const Element sh_10_9 = Element(std::sqrt(399.0 / 99.0)) * y * sh_9_8 - Element(std::sqrt(1680.0 / 1683.0)) * radius2 * sh_8_7;
const Element sh_10_10 = Element(std::sqrt(399.0 / 100.0)) * y * sh_9_9 - Element(std::sqrt(1701.0 / 1700.0)) * radius2 * sh_8_8;
const Element sh_10_11 = Element(std::sqrt(399.0 / 99.0)) * y * sh_9_10 - Element(std::sqrt(1680.0 / 1683.0)) * radius2 * sh_8_9;
const Element sh_10_12 = Element(std::sqrt(399.0 / 96.0)) * y * sh_9_11 - Element(std::sqrt(1617.0 / 1632.0)) * radius2 * sh_8_10;
const Element sh_10_13 = Element(std::sqrt(399.0 / 91.0)) * y * sh_9_12 - Element(std::sqrt(1512.0 / 1547.0)) * radius2 * sh_8_11;
const Element sh_10_14 = Element(std::sqrt(399.0 / 84.0)) * y * sh_9_13 - Element(std::sqrt(1365.0 / 1428.0)) * radius2 * sh_8_12;
const Element sh_10_15 = Element(std::sqrt(399.0 / 75.0)) * y * sh_9_14 - Element(std::sqrt(1176.0 / 1275.0)) * radius2 * sh_8_13;
const Element sh_10_16 = Element(std::sqrt(399.0 / 64.0)) * y * sh_9_15 - Element(std::sqrt(945.0 / 1088.0)) * radius2 * sh_8_14;
const Element sh_10_17 = Element(std::sqrt(399.0 / 51.0)) * y * sh_9_16 - Element(std::sqrt(672.0 / 867.0)) * radius2 * sh_8_15;
const Element sh_10_18 = Element(std::sqrt(399.0 / 36.0)) * y * sh_9_17 - Element(std::sqrt(357.0 / 612.0)) * radius2 * sh_8_16;
const Element sh_10_19 = Element(std::sqrt(21.0)) * y * sh_9_18;
const Element sh_10_20 = Element(std::sqrt(21.0 / 20.0)) * (sh_9_18 * z - sh_9_0 * x);

// Degree 11
const Element sh_11_0 = Element(std::sqrt(23.0 / 22.0)) * (sh_10_0 * z + sh_10_20 * x);
const Element sh_11_1 = Element(std::sqrt(23.0)) * y * sh_10_0;
const Element sh_11_2 = Element(std::sqrt(483.0 / 40.0)) * y * sh_10_1 - Element(std::sqrt(437.0 / 760.0)) * radius2 * sh_9_0;
const Element sh_11_3 = Element(std::sqrt(483.0 / 57.0)) * y * sh_10_2 - Element(std::sqrt(828.0 / 1083.0)) * radius2 * sh_9_1;
const Element sh_11_4 = Element(std::sqrt(483.0 / 72.0)) * y * sh_10_3 - Element(std::sqrt(1173.0 / 1368.0)) * radius2 * sh_9_2;
const Element sh_11_5 = Element(std::sqrt(483.0 / 85.0)) * y * sh_10_4 - Element(std::sqrt(1472.0 / 1615.0)) * radius2 * sh_9_3;
const Element sh_11_6 = Element(std::sqrt(483.0 / 96.0)) * y * sh_10_5 - Element(std::sqrt(1725.0 / 1824.0)) * radius2 * sh_9_4;
const Element sh_11_7 = Element(std::sqrt(483.0 / 105.0)) * y * sh_10_6 - Element(std::sqrt(1932.0 / 1995.0)) * radius2 * sh_9_5;
const Element sh_11_8 = Element(std::sqrt(483.0 / 112.0)) * y * sh_10_7 - Element(std::sqrt(2093.0 / 2128.0)) * radius2 * sh_9_6;
const Element sh_11_9 = Element(std::sqrt(483.0 / 117.0)) * y * sh_10_8 - Element(std::sqrt(2208.0 / 2223.0)) * radius2 * sh_9_7;
const Element sh_11_10 = Element(std::sqrt(483.0 / 120.0)) * y * sh_10_9 - Element(std::sqrt(2277.0 / 2280.0)) * radius2 * sh_9_8;
const Element sh_11_11 = Element(std::sqrt(483.0 / 121.0)) * y * sh_10_10 - Element(std::sqrt(2300.0 / 2299.0)) * radius2 * sh_9_9;
const Element sh_11_12 = Element(std::sqrt(483.0 / 120.0)) * y * sh_10_11 - Element(std::sqrt(2277.0 / 2280.0)) * radius2 * sh_9_10;
const Element sh_11_13 = Element(std::sqrt(483.0 / 117.0)) * y * sh_10_12 - Element(std::sqrt(2208.0 / 2223.0)) * radius2 * sh_9_11;
const Element sh_11_14 = Element(std::sqrt(483.0 / 112.0)) * y * sh_10_13 - Element(std::sqrt(2093.0 / 2128.0)) * radius2 * sh_9_12;
const Element sh_11_15 = Element(std::sqrt(483.0 / 105.0)) * y * sh_10_14 - Element(std::sqrt(1932.0 / 1995.0)) * radius2 * sh_9_13;
const Element sh_11_16 = Element(std::sqrt(483.0 / 96.0)) * y * sh_10_15 - Element(std::sqrt(1725.0 / 1824.0)) * radius2 * sh_9_14;
const Element sh_11_17 = Element(std::sqrt(483.0 / 85.0)) * y * sh_10_16 - Element(std::sqrt(1472.0 / 1615.0)) * radius2 * sh_9_15;
const Element sh_11_18 = Element(std::sqrt(483.0 / 72.0)) * y * sh_10_17 - Element(std::sqrt(1173.0 / 1368.0)) * radius2 * sh_9_16;
const Element sh_11_19 = Element(std::sqrt(483.0 / 57.0)) * y * sh_10_18 - Element(std::sqrt(828.0 / 1083.0)) * radius2 * sh_9_17;
const Element sh_11_20 = Element(std::sqrt(483.0 / 40.0)) * y * sh_10_19 - Element(std::sqrt(437.0 / 760.0)) * radius2 * sh_9_18;
const Element sh_11_21 = Element(std::sqrt(23.0)) * y * sh_10_20;
const Element sh_11_22 = Element(std::sqrt(23.0 / 22.0)) * (sh_10_20 * z - sh_10_0 * x);

// Degree 12
const Element sh_12_0 = Element(std::sqrt(25.0 / 24.0)) * (sh_11_0 * z + sh_11_22 * x);
const Element sh_12_1 = Element(std::sqrt(25.0)) * y * sh_11_0;
const Element sh_12_2 = Element(std::sqrt(575.0 / 44.0)) * y * sh_11_1 - Element(std::sqrt(525.0 / 924.0)) * radius2 * sh_10_0;
const Element sh_12_3 = Element(std::sqrt(575.0 / 63.0)) * y * sh_11_2 - Element(std::sqrt(1000.0 / 1323.0)) * radius2 * sh_10_1;
const Element sh_12_4 = Element(std::sqrt(575.0 / 80.0)) * y * sh_11_3 - Element(std::sqrt(1425.0 / 1680.0)) * radius2 * sh_10_2;
const Element sh_12_5 = Element(std::sqrt(575.0 / 95.0)) * y * sh_11_4 - Element(std::sqrt(1800.0 / 1995.0)) * radius2 * sh_10_3;
const Element sh_12_6 = Element(std::sqrt(575.0 / 108.0)) * y * sh_11_5 - Element(std::sqrt(2125.0 / 2268.0)) * radius2 * sh_10_4;
const Element sh_12_7 = Element(std::sqrt(575.0 / 119.0)) * y * sh_11_6 - Element(std::sqrt(2400.0 / 2499.0)) * radius2 * sh_10_5;
const Element sh_12_8 = Element(std::sqrt(575.0 / 128.0)) * y * sh_11_7 - Element(std::sqrt(2625.0 / 2688.0)) * radius2 * sh_10_6;
const Element sh_12_9 = Element(std::sqrt(575.0 / 135.0)) * y * sh_11_8 - Element(std::sqrt(2800.0 / 2835.0)) * radius2 * sh_10_7;
const Element sh_12_10 = Element(std::sqrt(575.0 / 140.0)) * y * sh_11_9 - Element(std::sqrt(2925.0 / 2940.0)) * radius2 * sh_10_8;
const Element sh_12_11 = Element(std::sqrt(575.0 / 143.0)) * y * sh_11_10 - Element(std::sqrt(3000.0 / 3003.0)) * radius2 * sh_10_9;
const Element sh_12_12 = Element(std::sqrt(575.0 / 144.0)) * y * sh_11_11 - Element(std::sqrt(3025.0 / 3024.0)) * radius2 * sh_10_10;
const Element sh_12_13 = Element(std::sqrt(575.0 / 143.0)) * y * sh_11_12 - Element(std::sqrt(3000.0 / 3003.0)) * radius2 * sh_10_11;
const Element sh_12_14 = Element(std::sqrt(575.0 / 140.0)) * y * sh_11_13 - Element(std::sqrt(2925.0 / 2940.0)) * radius2 * sh_10_12;
const Element sh_12_15 = Element(std::sqrt(575.0 / 135.0)) * y * sh_11_14 - Element(std::sqrt(2800.0 / 2835.0)) * radius2 * sh_10_13;
const Element sh_12_16 = Element(std::sqrt(575.0 / 128.0)) * y * sh_11_15 - Element(std::sqrt(2625.0 / 2688.0)) * radius2 * sh_10_14;
const Element sh_12_17 = Element(std::sqrt(575.0 / 119.0)) * y * sh_11_16 - Element(std::sqrt(2400.0 / 2499.0)) * radius2 * sh_10_15;
const Element sh_12_18 = Element(std::sqrt(575.0 / 108.0)) * y * sh_11_17 - Element(std::sqrt(2125.0 / 2268.0)) * radius2 * sh_10_16;
const Element sh_12_19 = Element(std::sqrt(575.0 / 95.0)) * y * sh_11_18 - Element(std::sqrt(1800.0 / 1995.0)) * radius2 * sh_10_17;
const Element sh_12_20 = Element(std::sqrt(575.0 / 80.0)) * y * sh_11_19 - Element(std::sqrt(1425.0 / 1680.0)) * radius2 * sh_10_18;
const Element sh_12_21 = Element(std::sqrt(575.0 / 63.0)) * y * sh_11_20 - Element(std::sqrt(1000.0 / 1323.0)) * radius2 * sh_10_19;
const Element sh_12_22 = Element(std::sqrt(575.0 / 44.0)) * y * sh_11_21 - Element(std::sqrt(525.0 / 924.0)) * radius2 * sh_10_20;
const Element sh_12_23 = Element(std::sqrt(25.0)) * y * sh_11_22;
const Element sh_12_24 = Element(std::sqrt(25.0 / 24.0)) * (sh_11_22 * z - sh_11_0 * x);
"""

NORMALIZE_COORDINATES = r"""
const Element length = vector_norm(x, y, z);
const Element denominator = clamp_min(length, Element(1e-12));
const Element inverse_length = Element(1.0) / denominator;
const Element nx = x * inverse_length;
const Element ny = y * inverse_length;
const Element nz = z * inverse_length;
"""
SCALAR_KERNEL = r"""{layout_declaration}__global__ __launch_bounds__({block_threads},{min_blocks}) void {kernel}({parameters},int64_t num_items{layout_parameter}) {{
    __shared__ Element output_tile[{warps}][{tile_columns}*33];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for(Index block_begin=Index(blockIdx.x)*{block_threads};block_begin<num_items;block_begin+=Index(gridDim.x)*{block_threads}) {{
        const Index row = block_begin + threadIdx.x;
        const Index item = row < num_items ? row : Index(num_items - 1);
{item_setup}{expressions}
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
    """Evaluate the bundled expressions in their declared operation order.

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

    # Contract a multiply consumed only by its add/subtract into native FP32 FMA.
    # Keeping shared multiplies avoids recomputation and preserves their reuse.
    uses = {}
    for node in nodes:
        for arg in node.args:
            uses[arg.index] = uses.get(arg.index, 0) + 1
    for node in selected:
        uses[node.index] = uses.get(node.index, 0) + 1
    fused = {}
    if not is_double:
        for node in nodes:
            if node.op in ('add', 'sub'):
                for side, arg in enumerate(node.args):
                    if arg.op == 'mul' and uses.get(arg.index) == 1:
                        fused[node.index] = (side, arg)
                        break
    skipped = {arg.index for _, arg in fused.values()}
    assignments = []
    for n in nodes:
        if n.op == 'constant' or n.index in skipped:
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
            if n.index in fused:
                side, multiply = fused[n.index]
                left, right = (
                    float_literal(float.fromhex(arg.value), meta['dtype'])
                    if arg.op == 'constant' else f'v{arg.index}'
                    for arg in multiply.args
                )
                other = a[1 - side]
                if n.op == 'sub':
                    if side == 0:
                        other = f'(-{other})'
                    else:
                        left = f'(-{left})'
                value = f'__fmaf_rn({left},{right},{other})'
            else:
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
        assignments.append((n.index, f'const {type_} v{n.index}={value};'))
    groups = []
    for name in names:
        values = outputs[name]
        for begin in range(0, len(values), OUTPUT_TILE_COLUMNS):
            group = values[begin:begin+OUTPUT_TILE_COLUMNS]
            ready = max((value.index for value in group if value.op != 'constant'), default=-1)
            groups.append((ready, name, begin, group, len(values)))

    # Store each output when its arithmetic is ready, instead of keeping
    # every output live until the final scalar instruction.
    stores = {}
    previous_flush = -1
    output_unroll = '' if program['order'] == 0 else ' 4'
    for ready, name, begin, group, width in sorted(groups, key=lambda item: item[0]):
        flush = max(previous_flush, ready)
        for column, value in enumerate(group):
            constant = value.op == 'constant'
            at = max(previous_flush, -1 if constant else value.index)
            expression = float_literal(float.fromhex(value.value), meta['dtype']) if constant else f'v{value.index}'
            stores.setdefault(at, []).append(f'output_tile[warp][{column}*33+lane]={expression};')
        stores.setdefault(flush, []).extend([
            '__syncwarp();',
            (
                f'#pragma unroll{output_unroll}\n'
                f'for(int offset=lane;offset<32*{len(group)};offset+=32) {{\n'
                f'    const Index output_item=block_begin+warp*32+offset/{len(group)};\n'
                f'    if(output_item<num_items) out_{name}[output_item*{width}+{begin}+offset%{len(group)}]='
                f'output_tile[warp][(offset%{len(group)})*33+offset/{len(group)}];\n'
                f'}}'
            ),
            '__syncwarp();',
        ])
        previous_flush = flush
    lines = stores.pop(-1, [])
    for index, assignment in assignments:
        lines.append(assignment)
        lines.extend(stores.pop(index, ()))
    if stores:
        raise ValueError('Spherical-harmonic output scheduled before its scalar value')
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
            tile_columns=min(OUTPUT_TILE_COLUMNS, max(len(outputs[name]) for name in names)),
            block_threads=BLOCK_THREADS, warps=BLOCK_THREADS//32,
            min_blocks=TRIPLE_BACKWARD_MIN_BLOCKS if program['order'] == 3 and not is_double else 1,
            parameters=','.join(parameters), layout_parameter=layout_parameter,
            item_setup=item_setup, expressions=indent("\n".join(lines), "        "),
        )
        + HOST_BINDING.format(
            launch_parameters=','.join(f'torch::Tensor {role}' for role in program['inputs']),
            first=first, allocations="\n    ".join(allocations),
            layout_setup=layout_setup, kernel=kernel, arguments=','.join(arguments),
            layout_argument=layout_argument, results=','.join(returns),
            block_threads=BLOCK_THREADS, block_last=BLOCK_THREADS-1,
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
    bound = max(max(3, op.meta['out_dim']) * num_items, input_span) + 65535 * BLOCK_THREADS
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
    key = (0, None, ('x',), index_bits, arch, norm_layout, x.ndim if direct_input else None)
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
            source = combine_cuda_sources({
                'forward': source,
                'backward': make_backward_source(meta, backward_program),
            })
            extension = load_cuda_extension(
                prefix.replace('forward', 'forward_backward'), source,
                index_bits=index_bits, arch=arch, use_fast_math=False, minimal_headers=True,
            )
            launch = extension.forward.launch
            backward_bound = max(3, meta['out_dim']) * num_items + 65535 * BLOCK_THREADS
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
                backward_key = (1, None, available, backward_bits, arch, norm_layout)
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
