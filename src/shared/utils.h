#ifndef PPKMHDH_UTILS_H
#define PPKMHDH_UTILS_H

#include <cmath>
#include <iosfwd>

#define THRESHOLD 1e-12

#define ISFUZZYNULL(a) (std::abs(a) < THRESHOLD)
#define FUZZYCOMPARE(a, b) \
    ((ISFUZZYNULL(a) && ISFUZZYNULL(b)) || \
     (std::abs((a) - (b)) * 1000000000000. <= std::fmin(std::abs(a), std::abs(b))))
#define FUZZYLIMITS(x, a, b) \
    (((x) > ((a) - THRESHOLD)) && ((x) < ((b) + THRESHOLD)))

void print_current_date();

#endif // PPKMHD_UTILS_H
