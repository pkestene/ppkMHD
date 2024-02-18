#include "mood/Matrix.h"

#include <lapacke.h> // for LAPACKE_dgeqrf
#include <cblas.h>   // for cblas_

#include "shared/utils.h"

namespace mood
{

// =======================================================
// =======================================================
void
Matrix::compute_qr()
{

  // leading dimension of this (assuming row-major storage)
  lapack_int lda = n;

  lapack_int min_mn = m < n ? m : n;
  double *   tau = (double *)malloc(min_mn * sizeof(double));

  // this->_data is modified here by the QR decomposition
  lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, data(), lda, tau);
  UNUSED(info);

  free(tau);

} // compute_qr

// =======================================================
// =======================================================
/**
 *
 */
void
compute_pseudo_inverse(const Matrix & A, Matrix & A_PI)
{

  const int m = A.m;
  const int n = A.n;

  /*
   * here we assume A.m > A.n (more rows than columns).
   *
   * A_PI must/will be square matrix of sizes m,m.
   */

  // if A_PI has wrong sizes, we re-allocate it
  if (A_PI.m != m and A_PI.n != m)
    A_PI.allocate(m, m);

  // create a working copy of A
  Matrix R(A);

  // R will contain the upper triangular factor of QR decomposition
  R.compute_qr();

  // transpose A
  Matrix AT(A);
  AT.transpose();

  // linear solve of R^T * X = A^T, i.e. compute X = (R^T)^(-1) * A^T,
  cblas_dtrsm(CblasRowMajor,
              CblasLeft,
              CblasUpper,
              CblasTrans,
              CblasNonUnit,
              n,
              m,
              1.0,
              R.data(),
              R.ld(),
              AT.data(),
              AT.ld());

  // linear solve R * X=(R^T)^-1*A^T, i.e compute X = (R^-1)*(R^T)^(-1) * A^T
  cblas_dtrsm(CblasRowMajor,
              CblasLeft,
              CblasUpper,
              CblasNoTrans,
              CblasNonUnit,
              n,
              m,
              1.0,
              R.data(),
              R.ld(),
              AT.data(),
              AT.ld());

  // here is the pseudo-inverse
  A_PI = AT;

} // compute_pseudo_inverse

} // namespace mood
