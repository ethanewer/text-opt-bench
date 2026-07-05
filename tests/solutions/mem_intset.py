"""Reference: sorted 32-bit array + binary search. ~4 bytes/element vs the
baseline set's hash slot + boxed int per element."""

from array import array


def build(ints):
    a = array("i", sorted(ints))
    return a


def contains(index, x):
    lo, hi = 0, len(index)
    while lo < hi:
        mid = (lo + hi) >> 1
        if index[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(index) and index[lo] == x
