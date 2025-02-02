#include <math.h>
#include <iostream>

#include "GeometricTerms.h"
#include "Binomial.h"

namespace mood
{

// =======================================================
// ==== Class GeometricTerms IMPL ========================
// =======================================================


// =======================================================
// =======================================================
GeometricTerms::GeometricTerms(real_t dx, real_t dy, real_t dz)
  : dx(dx)
  , dy(dy)
  , dz(dz)
{

  // assumes that stencil_radius is in [1,6]
  // so we just need to instantiate the maximum value

} // GeometricTerms::GeometricTerms


// =======================================================
// =======================================================
GeometricTerms::~GeometricTerms() {} // GeometricTerms::~GeometricTerms

// =======================================================
// =======================================================
real_t
GeometricTerms::eval_moment(int i, int j, int n, int m)
{

  return 1.0 / dx / dy * pow(dx, n + 1) / (n + 1) * (pow(i + 0.5, n + 1) - pow(i - 0.5, n + 1)) *
         pow(dy, m + 1) / (m + 1) * (pow(j + 0.5, m + 1) - pow(j - 0.5, m + 1));

} // GeometricTerms::eval_moment

// =======================================================
// =======================================================
real_t
GeometricTerms::eval_moment(int i, int j, int k, int n, int m, int l)
{

  return 1.0 / dx / dy / dz * pow(dx, n + 1) / (n + 1) *
         (pow(i + 0.5, n + 1) - pow(i - 0.5, n + 1)) * pow(dy, m + 1) / (m + 1) *
         (pow(j + 0.5, m + 1) - pow(j - 0.5, m + 1)) * pow(dz, l + 1) / (l + 1) *
         (pow(k + 0.5, l + 1) - pow(k - 0.5, l + 1));

} // GeometricTerms::eval_moment

// =======================================================
// =======================================================
real_t
GeometricTerms::eval_hat(int i, int j, int n, int m)
{

  return eval_moment(i, j, n, m) - eval_moment(0, 0, n, m);

} // GeometricTerms::eval_hat

// =======================================================
// =======================================================
real_t
GeometricTerms::eval_hat(int i, int j, int k, int n, int m, int l)
{

  return eval_moment(i, j, k, n, m, l) - eval_moment(0, 0, 0, n, m, l);

} // GeometricTerms::eval_hat

} // namespace mood
