#ifndef MOOD_POLYNOMIAL_RECONSTRUCTION_FUNCTORS_H_
#define MOOD_POLYNOMIAL_RECONSTRUCTION_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"
#include "shared/RiemannSolvers.h"

#include "mood/mood_shared.h"
#include "mood/Polynomial.h"
#include "mood/MoodBaseFunctor.h"
#include "mood/QuadratureRules.h"

namespace mood
{

// =======================================================================
// =======================================================================
/**
 * Compute MOOD polynomial coefficients.
 *
 * The functor simply load data from neighbor cells that belong to the given
 * stencil and compute for the central cell, an interpolating polynomial in
 * a least-square sense, i.e. the polynomial coefficients are the solution
 * of a linear system, for which the matrix is purely geometric, and solved
 * by using the pseudo-inverse (independent from the cell location in a regular
 * cartesian mesh).
 *
 * Stencil has been carefully chosen to have a number of cells sufficiently large
 * so that the least-square system can be solved. If the stencil size is low,
 * one can notice that the pseudo inverse can not be computer using the QR
 * decomposition.
 *
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 *
 * stencilId must be known at compile time, so that stencilSize is too.
 */
template <int dim, int degree, STENCIL_ID stencilId>
class ComputeReconstructionPolynomialFunctor : public MoodBaseFunctor<dim, degree>
{

public:
  //! the actual typedef is defined in the base class
  using typename MoodBaseFunctor<dim, degree>::DataArray;

  //! the actual typedef is defined in the base class
  using typename PolynomialEvaluator<dim, degree>::coefs_t;

  //! total number of coefficients in the reconstructing polynomial
  static const int ncoefs = mood::binomial<dim + degree, dim>();

  using MonomMap = typename mood::MonomialMap<dim, degree>::MonomMap;


  /**
   * Constructor for 2D/3D.
   *
   * \param[in] Udata array of conservative variables
   * \param[out] polyCoefs array of DataArray, one for each polynomial coefficient
   * \param[in] params (for isize, jsize, ...)
   * \param[in] stencil (array containing neighbor x,y,z coordinates)
   * \param[in] mat_pi pseudo-inverse of the geometric terms matrix.
   */
  ComputeReconstructionPolynomialFunctor(HydroParams                      params,
                                         MonomMap                         monomMap,
                                         DataArray                        Udata,
                                         Kokkos::Array<DataArray, ncoefs> polyCoefs,
                                         Stencil                          stencil,
                                         mood_matrix_pi_t                 mat_pi)
    : MoodBaseFunctor<dim, degree>(params, monomMap)
    , Udata(Udata)
    , polyCoefs(polyCoefs)
    , stencil(stencil)
    , mat_pi(mat_pi){};

  ~ComputeReconstructionPolynomialFunctor(){};

  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    // const real_t dx = this->params.dx;
    // const real_t dy = this->params.dy;

    const real_t nbvar = this->params.nbvar;

    int i, j;
    index2coord(index, i, j, isize, jsize);

    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t, stencil_size - 1> rhs;

    if (j >= ghostWidth - 1 && j < jsize - ghostWidth + 1 && i >= ghostWidth - 1 &&
        i < isize - ghostWidth + 1)
    {

      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // retrieve neighbors data for variable ivar, and build rhs
        int irhs = 0;
        for (int is = 0; is < stencil_size; ++is)
        {
          int x = stencil.offsets(is, 0);
          int y = stencil.offsets(is, 1);
          if (x != 0 or y != 0)
          {
            rhs[irhs] = Udata(i + x, j + y, ivar) - Udata(i, j, ivar);
            irhs++;
          }
        } // end for is

        // retrieve reconstruction polynomial coefficients in current cell
        coefs_t coefs_c;
        coefs_c[0] = Udata(i, j, ivar);
        for (int icoef = 0; icoef < mat_pi.extent(0); ++icoef)
        {
          real_t tmp = 0;
          for (int ik = 0; ik < mat_pi.extent(1); ++ik)
          {
            tmp += mat_pi(icoef, ik) * rhs[ik];
          }
          coefs_c[icoef + 1] = tmp;
        }

        // copy back results on device memory
        for (int icoef = 0; icoef < ncoefs; ++icoef)
          polyCoefs[icoef](i, j, ivar) = coefs_c[icoef];

      } // end for ivar

    } // end if

  } // end functor 2d

  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    // const real_t dx = this->params.dx;
    // const real_t dy = this->params.dy;
    // const real_t dz = this->params.dz;

    const real_t nbvar = this->params.nbvar;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t, stencil_size - 1> rhs;

    if (k >= ghostWidth && k < ksize - ghostWidth + 1 && j >= ghostWidth &&
        j < jsize - ghostWidth + 1 && i >= ghostWidth && i < isize - ghostWidth + 1)
    {

      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // retrieve neighbors data for ivar, and build rhs
        int irhs = 0;
        for (int is = 0; is < stencil_size; ++is)
        {
          int x = stencil.offsets(is, 0);
          int y = stencil.offsets(is, 1);
          int z = stencil.offsets(is, 2);
          if (x != 0 or y != 0 or z != 0)
          {
            rhs[irhs] = Udata(i + x, j + y, k + z, ivar) - Udata(i, j, k, ivar);
            irhs++;
          }
        } // end for is

        // retrieve reconstruction polynomial coefficients in current cell
        coefs_t coefs_c;
        coefs_c[0] = Udata(i, j, k, ivar);
        for (int icoef = 0; icoef < mat_pi.extent(0); ++icoef)
        {
          real_t tmp = 0;
          for (int ik = 0; ik < mat_pi.extent(1); ++ik)
          {
            tmp += mat_pi(icoef, ik) * rhs[ik];
          }
          coefs_c[icoef + 1] = tmp;
        }

        // copy back results on device memory
        for (int icoef = 0; icoef < ncoefs; ++icoef)
          polyCoefs[icoef](i, j, k, ivar) = coefs_c[icoef];

      } // end for ivar

    } // end if i,j,k

  } // end functor 3d

  DataArray                        Udata;
  Kokkos::Array<DataArray, ncoefs> polyCoefs;

  Stencil          stencil;
  mood_matrix_pi_t mat_pi;

  // get the number of cells in stencil
  static constexpr int stencil_size = STENCIL_SIZE[stencilId];

  // get the number of quadrature point per face corresponding to this stencil
  static constexpr int nbQuadPts = QUADRATURE_NUM_POINTS[stencilId];

}; // class ComputeReconstructionPolynomialFunctor

} // namespace mood

#endif // MOOD_POLYNOMIAL_RECONSTRUCTION_FUNCTORS_H_
