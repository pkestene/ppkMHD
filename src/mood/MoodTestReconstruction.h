#ifndef MOOD_TEST_RECONSTRUCTION_H_
#define MOOD_TEST_RECONSTRUCTION_H_

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
 * Compute MOOD fluxes.
 *
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 * - FluxData_z may or may not be allocated (depending dim==2 or 3).
 *
 * stencilId must be known at compile time, so that stencilSize is too.
 */
template <int dim, int degree, STENCIL_ID stencilId>
class TestReconstructionFunctor : public MoodBaseFunctor<dim, degree>
{

public:
  using typename MoodBaseFunctor<dim, degree>::DataArray;
  using typename MoodBaseFunctor<dim, degree>::HydroState;
  using typename PolynomialEvaluator<dim, degree>::coefs_t;

  //! total number of coefficients in the polynomial
  static const int ncoefs = mood::binomial<dim + degree, dim>();

  /**
   * Constructor for 2D/3D.
   */
  TestReconstructionFunctor(DataArray                        Udata,
                            Kokkos::Array<DataArray, ncoefs> polyCoefs,
                            DataArray                        RecState1,
                            DataArray                        RecState2,
                            DataArray                        RecState3,
                            HydroParams                      params,
                            Stencil                          stencil,
                            mood_matrix_pi_t                 mat_pi,
                            QuadLoc_2d_t                     QUAD_LOC_2D)
    : MoodBaseFunctor<dim, degree>(params)
    , Udata(Udata)
    , polyCoefs(polyCoefs)
    , RecState1(RecState1)
    , RecState2(RecState2)
    , RecState3(RecState3)
    , stencil(stencil)
    , mat_pi(mat_pi)
    , QUAD_LOC_2D(QUAD_LOC_2D){};

  ~TestReconstructionFunctor(){};

  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    const real_t nbvar = this->params.nbvar;

    // riemann solver states left/right (conservative variables),
    // one for each quadrature point
    HydroState UL[nbQuadPts], UR[nbQuadPts];

    // primitive variables left / right states
    HydroState qL, qR, qgdnv;
    real_t     c;

    // accumulate flux over all quadrature points
    HydroState rec1, rec2;

    int i, j;
    index2coord(index, i, j, isize, jsize);

    /*********************
     * along DIR_X
     *********************/
    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth + 1)
    {

      // for each variable,
      // retrieve reconstruction polynomial coefficients in current cell
      // and all compute UL / UR states
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // current cell
        coefs_t coefs_c;

        // neighbor cell
        coefs_t coefs_n;

        // read polynomial coefficients
        for (int icoef = 0; icoef < ncoefs; ++icoef)
        {
          coefs_c[icoef] = polyCoefs[icoef](i, j, ivar);
          coefs_n[icoef] = polyCoefs[icoef](i - 1, j, ivar);
        }

        // reconstruct Udata on the left face along X direction
        // for each quadrature points
        real_t x, y;
        for (int iq = 0; iq < nbQuadPts; ++iq)
        {

          // left  interface in neighbor cell
          x = QUAD_LOC_2D(nbQuadPts - 1, DIR_X, FACE_MAX, iq, IX);
          y = QUAD_LOC_2D(nbQuadPts - 1, DIR_X, FACE_MAX, iq, IY);
          UL[iq][ivar] = this->eval(x * dx, y * dy, coefs_n);

          // right interface in current cell
          x = QUAD_LOC_2D(nbQuadPts - 1, DIR_X, FACE_MIN, iq, IX);
          y = QUAD_LOC_2D(nbQuadPts - 1, DIR_X, FACE_MIN, iq, IY);
          UR[iq][ivar] = this->eval(x * dx, y * dy, coefs_c);
        }

      } // end for ivar

      // check if the reconstructed states are valid, if not we use  Udata
      for (int iq = 0; iq < nbQuadPts; ++iq)
      {

        if (this->isValid(UL[iq]) == 0)
        {
          // change UL into Udata from neighbor
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UL[iq][ivar] = polyCoefs[0](i - 1, j, ivar);
        }

        if (this->isValid(UR[iq]) == 0)
        {
          // change UR into Udata from current cell
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UR[iq][ivar] = polyCoefs[0](i, j, ivar);
        }

      } // end check validity

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {
        RecState1(i, j, ivar) = UL[0][ivar];
        RecState2(i, j, ivar) = UR[0][ivar];
      }

    } // end if

    /*********************
     * flux along DIR_Y
     *********************/
    // if(j >= ghostWidth && j < jsize-ghostWidth+1   &&
    //    i >= ghostWidth && i < isize-ghostWidth ) {

    //   // for each variable,
    //   // retrieve reconstruction polynomial coefficients in current cell
    //   // and all compute UL / UR states
    //   for (int ivar=0; ivar<nbvar; ++ivar) {

    // 	// current cell
    // 	coefs_t coefs_c;

    // 	// neighbor cell
    // 	coefs_t coefs_n;

    // 	// read polynomial coefficients
    // 	for (int icoef=0; icoef<ncoefs; ++icoef) {
    // 	  coefs_c[icoef] = polyCoefs[icoef](i  ,j  ,ivar);
    // 	  coefs_n[icoef] = polyCoefs[icoef](i  ,j-1,ivar);
    // 	}

    // 	// reconstruct Udata on the left face along X direction
    // 	// for each quadrature points
    // 	real_t x,y;
    // 	for (int iq = 0; iq<nbQuadPts; ++iq) {

    // 	  // left  interface in neighbor cell
    // 	  x = QUAD_LOC_2D(nbQuadPts-1,DIR_Y,FACE_MAX,iq,IX);
    // 	  y = QUAD_LOC_2D(nbQuadPts-1,DIR_Y,FACE_MAX,iq,IY);
    // 	  UL[iq][ivar] = this->eval(x*dx, y*dy, coefs_n);

    // 	  // right interface in current cell
    // 	  x = QUAD_LOC_2D(nbQuadPts-1,DIR_Y,FACE_MIN,iq,IX);
    // 	  y = QUAD_LOC_2D(nbQuadPts-1,DIR_Y,FACE_MIN,iq,IY);
    // 	  UR[iq][ivar] = this->eval(x*dx, y*dy, coefs_c);

    // 	}

    //   } // end for ivar

    // } // end if

  } // end functor 2d

  //! functor for 3d
  /************* UNFINISHED - TODO ***************/
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {} // end functor 3d

  DataArray                        Udata;
  Kokkos::Array<DataArray, ncoefs> polyCoefs;
  DataArray                        RecState1, RecState2, RecState3;

  Stencil          stencil;
  mood_matrix_pi_t mat_pi;
  QuadLoc_2d_t     QUAD_LOC_2D;

  // get the number of cells in stencil
  static constexpr int stencil_size = STENCIL_SIZE[stencilId];

  // get the number of quadrature point per face corresponding to this stencil
  static constexpr int nbQuadPts = QUADRATURE_NUM_POINTS[stencilId];

}; // class TestReconstructionFunctor

} // namespace mood

#endif // MOOD_TEST_RECONSTRUCTION_H_
