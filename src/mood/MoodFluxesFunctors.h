#ifndef MOOD_FLUXES_FUNCTORS_H_
#define MOOD_FLUXES_FUNCTORS_H_

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
class ComputeFluxesFunctor : public MoodBaseFunctor<dim, degree>
{

public:
  using typename MoodBaseFunctor<dim, degree>::DataArray;
  using typename MoodBaseFunctor<dim, degree>::HydroState;
  using typename PolynomialEvaluator<dim, degree>::coefs_t;
  using MonomMap = typename mood::MonomialMap<dim, degree>::MonomMap;

  //! total number of coefficients in the polynomial
  static const int ncoefs = mood::binomial<dim + degree, dim>();

  /**
   * Constructor for 2D/3D.
   */
  ComputeFluxesFunctor(HydroParams                      params,
                       MonomMap                         monomMap,
                       DataArray                        Udata,
                       Kokkos::Array<DataArray, ncoefs> polyCoefs,
                       DataArray                        FluxData_x,
                       DataArray                        FluxData_y,
                       DataArray                        FluxData_z,
                       Stencil                          stencil,
                       mood_matrix_pi_t                 mat_pi,
                       QuadLoc_2d_t                     QUAD_LOC_2D,
                       QuadLoc_3d_t                     QUAD_LOC_3D,
                       real_t                           dtdx,
                       real_t                           dtdy,
                       real_t                           dtdz)
    : MoodBaseFunctor<dim, degree>(params, monomMap)
    , Udata(Udata)
    , polyCoefs(polyCoefs)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z)
    , stencil(stencil)
    , mat_pi(mat_pi)
    , QUAD_LOC_2D(QUAD_LOC_2D)
    , QUAD_LOC_3D(QUAD_LOC_3D)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  ~ComputeFluxesFunctor(){};

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

    // Quadrature weights when using 1 point (Gauss-Legendre).
    // const real_t QUADRATURE_WEIGHTS_N1[1] = {1.0};

    // Quadrature weights when using 2 points (Gauss-Legendre).
    const real_t QUADRATURE_WEIGHTS_N2[2] = { 0.5, 0.5 };

    // Quadrature weights when using 3 points (Gauss-Legendre).
    const real_t QUADRATURE_WEIGHTS_N3[3] = { 5.0 / 18, 8.0 / 18, 5.0 / 18 };

    // riemann solver states left/right (conservative variables),
    // one for each quadrature point
    HydroState UL[nbQuadPts], UR[nbQuadPts];

    // primitive variables left / right states
    HydroState qL, qR, qgdnv;
    real_t     c;

    // accumulate flux over all quadrature points
    HydroState flux, flux_tmp;

    int i, j;
    index2coord(index, i, j, isize, jsize);

    /*********************
     * flux along DIR_X
     *********************/
    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth + 1)
    {

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

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

        // if ( this->isValid(UL[iq]) == 0 ) {
        //   // change UL into Udata from neighbor
        //   for (int ivar=0; ivar<nbvar; ++ivar)
        //     UL[iq][ivar] = polyCoefs[0](i-1,j,ivar);
        // }

        // if ( this->isValid(UR[iq]) == 0 ) {
        //   // change UR into Udata from current cell
        //   for (int ivar=0; ivar<nbvar; ++ivar)
        //     UR[iq][ivar] = polyCoefs[0](i,j,ivar);
        // }

        if (this->isValid(UL[iq]) == 0 or this->isValid(UR[iq]) == 0)
        {
          // change UL into Udata from neighbor
          // change UR into Udata from current cell
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {
            UL[iq][ivar] = polyCoefs[0](i - 1, j, ivar);
            UR[iq][ivar] = polyCoefs[0](i, j, ivar);
          }
        }

      } // end check validity

      // we can now perform the riemann solvers for each quadrature point
      for (int iq = 0; iq < nbQuadPts; ++iq)
      {

        // convert to primitive variable before riemann solver
        this->computePrimitives(UL[iq], &c, qL);
        this->computePrimitives(UR[iq], &c, qR);

        // compute riemann flux
        ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux_tmp, this->params);

        // the following will be nicer when nvcc will accept constexpr array.
        if (nbQuadPts == 1)
        {

          // just copy flux_tmp into flux
          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] = flux_tmp[ivar];
        }
        else if (nbQuadPts == 2)
        {

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N2[iq];
        }
        else if (nbQuadPts == 3)
        {

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N3[iq];
        }
      }

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_x(i, j, ivar) = flux[ivar] * dtdx;

    } // end if

    /*********************
     * flux along DIR_Y
     *********************/
    if (j >= ghostWidth && j < jsize - ghostWidth + 1 && i >= ghostWidth && i < isize - ghostWidth)
    {

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

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
          coefs_n[icoef] = polyCoefs[icoef](i, j - 1, ivar);
        }

        // reconstruct Udata on the left face along X direction
        // for each quadrature points
        real_t x, y;
        for (int iq = 0; iq < nbQuadPts; ++iq)
        {

          // left  interface in neighbor cell
          x = QUAD_LOC_2D(nbQuadPts - 1, DIR_Y, FACE_MAX, iq, IX);
          y = QUAD_LOC_2D(nbQuadPts - 1, DIR_Y, FACE_MAX, iq, IY);
          UL[iq][ivar] = this->eval(x * dx, y * dy, coefs_n);

          // right interface in current cell
          x = QUAD_LOC_2D(nbQuadPts - 1, DIR_Y, FACE_MIN, iq, IX);
          y = QUAD_LOC_2D(nbQuadPts - 1, DIR_Y, FACE_MIN, iq, IY);
          UR[iq][ivar] = this->eval(x * dx, y * dy, coefs_c);
        }

      } // end for ivar

      // check if the reconstructed states are valid, if not we use  Udata
      for (int iq = 0; iq < nbQuadPts; ++iq)
      {

        // if ( this->isValid(UL[iq]) == 0 ) {
        //   // change UL into Udata from neighbor
        //   for (int ivar=0; ivar<nbvar; ++ivar)
        //     UL[iq][ivar] = polyCoefs[0](i,j-1,ivar);
        // }

        // if ( this->isValid(UR[iq]) == 0 ) {
        //   // change UR into Udata from current cell
        //   for (int ivar=0; ivar<nbvar; ++ivar)
        //     UR[iq][ivar] = polyCoefs[0](i,j,ivar);
        // }

        if (this->isValid(UL[iq]) == 0 or this->isValid(UR[iq]) == 0)
        {
          // change UL into Udata from neighbor
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {
            UL[iq][ivar] = polyCoefs[0](i, j - 1, ivar);
            UR[iq][ivar] = polyCoefs[0](i, j, ivar);
          }
        }

      } // end check validity

      // we can now perform the riemann solvers for each quadrature point
      for (int iq = 0; iq < nbQuadPts; ++iq)
      {

        // convert to primitive variable before riemann solver
        this->computePrimitives(UL[iq], &c, qL);
        this->computePrimitives(UR[iq], &c, qR);

        // compute riemann flux
        // swap IU and IV velocity
        this->swap(qL[IU], qL[IV]);
        this->swap(qR[IU], qR[IV]);
        ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux_tmp, this->params);

        // the following will be nicer when nvcc will accept constexpr array.
        if (nbQuadPts == 1)
        {

          // just copy flux_tmp into flux
          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] = flux_tmp[ivar];
        }
        else if (nbQuadPts == 2)
        {

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N2[iq];
        }
        else if (nbQuadPts == 3)
        {

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N3[iq];
        }
      }

      // swap again IU and IV
      this->swap(flux[IU], flux[IV]);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_y(i, j, ivar) = flux[ivar] * dtdy;

    } // end if

  } // end functor 2d

  //! functor for 3d
  /************* UNFINISHED - TODO ***************/
  /************* UNFINISHED - TODO ***************/
  /************* UNFINISHED - TODO ***************/
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    const real_t nbvar = this->params.nbvar;

    // Quadrature weights when using 1 point (Gauss-Legendre).
    // const real_t QUADRATURE_WEIGHTS_N1[1] = {1.0};

    // Quadrature weights when using 2 points (Gauss-Legendre).
    const real_t QUADRATURE_WEIGHTS_N2[2] = { 0.5, 0.5 };

    // Quadrature weights when using 3 points (Gauss-Legendre).
    const real_t QUADRATURE_WEIGHTS_N3[3] = { 5.0 / 18, 8.0 / 18, 5.0 / 18 };

    // riemann solver states left/right (conservative variables),
    // one for each quadrature point
    HydroState UL[nbQuadPts3d], UR[nbQuadPts3d];

    // primitive variables left / right states
    HydroState qL, qR, qgdnv;
    real_t     c;

    // accumulate flux over all quadrature points
    HydroState flux, flux_tmp;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    /*********************
     * flux along DIR_X
     *********************/
    if (k >= ghostWidth && k < ksize - ghostWidth && j >= ghostWidth && j < jsize - ghostWidth &&
        i >= ghostWidth && i < isize - ghostWidth + 1)
    {

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

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
          coefs_c[icoef] = polyCoefs[icoef](i, j, k, ivar);
          coefs_n[icoef] = polyCoefs[icoef](i - 1, j, k, ivar);
        }

        // reconstruct Udata on the left face along X direction
        // for each quadrature points
        real_t x, y, z;
        for (int iq = 0; iq < nbQuadPts3d; ++iq)
        {

          // left  interface in neighbor cell
          x = QUAD_LOC_3D(nbQuadPts - 1, DIR_X, FACE_MAX, iq, IX);
          y = QUAD_LOC_3D(nbQuadPts - 1, DIR_X, FACE_MAX, iq, IY);
          z = QUAD_LOC_3D(nbQuadPts - 1, DIR_X, FACE_MAX, iq, IZ);
          UL[iq][ivar] = this->eval(x * dx, y * dy, z * dz, coefs_n);

          // right interface in current cell
          x = QUAD_LOC_3D(nbQuadPts - 1, DIR_X, FACE_MIN, iq, IX);
          y = QUAD_LOC_3D(nbQuadPts - 1, DIR_X, FACE_MIN, iq, IY);
          z = QUAD_LOC_3D(nbQuadPts - 1, DIR_X, FACE_MIN, iq, IZ);
          UR[iq][ivar] = this->eval(x * dx, y * dy, z * dz, coefs_c);
        }

      } // end for ivar

      // check if the reconstructed states are valid, if not we use  Udata
      for (int iq = 0; iq < nbQuadPts3d; ++iq)
      {

        if (this->isValid(UL[iq]) == 0)
        {
          // change UL into Udata from neighbor
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UL[iq][ivar] = polyCoefs[0](i - 1, j, k, ivar);
        }

        if (this->isValid(UR[iq]) == 0)
        {
          // change UR into Udata from current cell
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UR[iq][ivar] = polyCoefs[0](i, j, k, ivar);
        }

      } // end check validity

      // we can now perform the riemann solvers for each quadrature point
      for (int iq = 0; iq < nbQuadPts3d; ++iq)
      {

        // convert to primitive variable before riemann solver
        this->computePrimitives(UL[iq], &c, qL);
        this->computePrimitives(UR[iq], &c, qR);

        // compute riemann flux
        ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux_tmp, this->params);

        // the following will be nicer when nvcc will accept constexpr array.
        if (nbQuadPts == 1)
        {

          // just copy flux_tmp into flux
          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] = flux_tmp[ivar];
        }
        else if (nbQuadPts == 2)
        {

          int iq1 = iq / nbQuadPts;
          int iq2 = iq - iq1 * nbQuadPts;

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N2[iq1] * QUADRATURE_WEIGHTS_N2[iq2];
        }
        else if (nbQuadPts == 3)
        {

          int iq1 = iq / nbQuadPts;
          int iq2 = iq - iq1 * nbQuadPts;

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N3[iq1] * QUADRATURE_WEIGHTS_N3[iq2];
        }
      }

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_x(i, j, k, ivar) = flux[ivar] * dtdx;

    } // end - flux along X

    /*********************
     * flux along DIR_Y
     *********************/
    if (k >= ghostWidth && k < ksize - ghostWidth && j >= ghostWidth &&
        j < jsize - ghostWidth + 1 && i >= ghostWidth && i < isize - ghostWidth)
    {

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

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
          coefs_c[icoef] = polyCoefs[icoef](i, j, k, ivar);
          coefs_n[icoef] = polyCoefs[icoef](i, j - 1, k, ivar);
        }

        // reconstruct Udata on the left face along X direction
        // for each quadrature points
        real_t x, y, z;
        for (int iq = 0; iq < nbQuadPts3d; ++iq)
        {

          // left  interface in neighbor cell
          x = QUAD_LOC_3D(nbQuadPts - 1, DIR_Y, FACE_MAX, iq, IX);
          y = QUAD_LOC_3D(nbQuadPts - 1, DIR_Y, FACE_MAX, iq, IY);
          z = QUAD_LOC_3D(nbQuadPts - 1, DIR_Y, FACE_MAX, iq, IZ);
          UL[iq][ivar] = this->eval(x * dx, y * dy, z * dz, coefs_n);

          // right interface in current cell
          x = QUAD_LOC_3D(nbQuadPts - 1, DIR_Y, FACE_MIN, iq, IX);
          y = QUAD_LOC_3D(nbQuadPts - 1, DIR_Y, FACE_MIN, iq, IY);
          z = QUAD_LOC_3D(nbQuadPts - 1, DIR_Y, FACE_MIN, iq, IZ);
          UR[iq][ivar] = this->eval(x * dx, y * dy, z * dz, coefs_c);
        }

      } // end for ivar

      // check if the reconstructed states are valid, if not we use  Udata
      for (int iq = 0; iq < nbQuadPts3d; ++iq)
      {

        if (this->isValid(UL[iq]) == 0)
        {
          // change UL into Udata from neighbor
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UL[iq][ivar] = polyCoefs[0](i, j - 1, k, ivar);
        }

        if (this->isValid(UR[iq]) == 0)
        {
          // change UR into Udata from current cell
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UR[iq][ivar] = polyCoefs[0](i, j, k, ivar);
        }

      } // end check validity

      // we can now perform the riemann solvers for each quadrature point
      for (int iq = 0; iq < nbQuadPts3d; ++iq)
      {

        // convert to primitive variable before riemann solver
        this->computePrimitives(UL[iq], &c, qL);
        this->computePrimitives(UR[iq], &c, qR);

        // compute riemann flux
        // swap IU and IV velocity
        this->swap(qL[IU], qL[IV]);
        this->swap(qR[IU], qR[IV]);
        ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux_tmp, this->params);

        // the following will be nicer when nvcc will accept constexpr array.
        if (nbQuadPts == 1)
        {

          // just copy flux_tmp into flux
          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] = flux_tmp[ivar];
        }
        else if (nbQuadPts == 2)
        {

          int iq1 = iq / nbQuadPts;
          int iq2 = iq - iq1 * nbQuadPts;

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N2[iq1] * QUADRATURE_WEIGHTS_N2[iq2];
        }
        else if (nbQuadPts == 3)
        {

          int iq1 = iq / nbQuadPts;
          int iq2 = iq - iq1 * nbQuadPts;

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N3[iq1] * QUADRATURE_WEIGHTS_N3[iq2];
        }
      }

      // swap again IU and IV
      this->swap(flux[IU], flux[IV]);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_y(i, j, k, ivar) = flux[ivar] * dtdy;

    } // end flux along Y

    /*********************
     * flux along DIR_Z
     *********************/
    if (k >= ghostWidth && k < ksize - ghostWidth + 1 && j >= ghostWidth &&
        j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

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
          coefs_c[icoef] = polyCoefs[icoef](i, j, k, ivar);
          coefs_n[icoef] = polyCoefs[icoef](i, j, k - 1, ivar);
        }

        // reconstruct Udata on the left face along X direction
        // for each quadrature points
        real_t x, y, z;
        for (int iq = 0; iq < nbQuadPts3d; ++iq)
        {

          // left  interface in neighbor cell
          x = QUAD_LOC_3D(nbQuadPts - 1, DIR_Z, FACE_MAX, iq, IX);
          y = QUAD_LOC_3D(nbQuadPts - 1, DIR_Z, FACE_MAX, iq, IY);
          z = QUAD_LOC_3D(nbQuadPts - 1, DIR_Z, FACE_MAX, iq, IZ);
          UL[iq][ivar] = this->eval(x * dx, y * dy, z * dz, coefs_n);

          // right interface in current cell
          x = QUAD_LOC_3D(nbQuadPts - 1, DIR_Z, FACE_MIN, iq, IX);
          y = QUAD_LOC_3D(nbQuadPts - 1, DIR_Z, FACE_MIN, iq, IY);
          z = QUAD_LOC_3D(nbQuadPts - 1, DIR_Z, FACE_MIN, iq, IZ);
          UR[iq][ivar] = this->eval(x * dx, y * dy, z * dz, coefs_c);
        }

      } // end for ivar

      // check if the reconstructed states are valid, if not we use  Udata
      for (int iq = 0; iq < nbQuadPts3d; ++iq)
      {

        if (this->isValid(UL[iq]) == 0)
        {
          // change UL into Udata from neighbor
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UL[iq][ivar] = polyCoefs[0](i, j, k - 1, ivar);
        }

        if (this->isValid(UR[iq]) == 0)
        {
          // change UR into Udata from current cell
          for (int ivar = 0; ivar < nbvar; ++ivar)
            UR[iq][ivar] = polyCoefs[0](i, j, k, ivar);
        }

      } // end check validity

      // we can now perform the riemann solvers for each quadrature point
      for (int iq = 0; iq < nbQuadPts3d; ++iq)
      {

        // convert to primitive variable before riemann solver
        this->computePrimitives(UL[iq], &c, qL);
        this->computePrimitives(UR[iq], &c, qR);

        // compute riemann flux
        // swap IU and IV velocity
        this->swap(qL[IU], qL[IW]);
        this->swap(qR[IU], qR[IW]);
        ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux_tmp, this->params);

        // the following will be nicer when nvcc will accept constexpr array.
        if (nbQuadPts == 1)
        {

          // just copy flux_tmp into flux
          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] = flux_tmp[ivar];
        }
        else if (nbQuadPts == 2)
        {

          int iq1 = iq / nbQuadPts;
          int iq2 = iq - iq1 * nbQuadPts;

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N2[iq1] * QUADRATURE_WEIGHTS_N2[iq2];
        }
        else if (nbQuadPts == 3)
        {

          int iq1 = iq / nbQuadPts;
          int iq2 = iq - iq1 * nbQuadPts;

          for (int ivar = 0; ivar < nbvar; ++ivar)
            flux[ivar] += flux_tmp[ivar] * QUADRATURE_WEIGHTS_N3[iq1] * QUADRATURE_WEIGHTS_N3[iq2];
        }
      }

      // swap again IU and IW
      this->swap(flux[IU], flux[IW]);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_z(i, j, k, ivar) = flux[ivar] * dtdy;

    } // end flux along Z

  } // end functor 3d

  DataArray                        Udata;
  Kokkos::Array<DataArray, ncoefs> polyCoefs;
  DataArray                        FluxData_x, FluxData_y, FluxData_z;

  Stencil          stencil;
  mood_matrix_pi_t mat_pi;
  QuadLoc_2d_t     QUAD_LOC_2D;
  QuadLoc_3d_t     QUAD_LOC_3D;
  real_t           dtdx, dtdy, dtdz;

  // get the number of cells in stencil
  static constexpr int stencil_size = STENCIL_SIZE[stencilId];

  // get the number of quadrature point per face corresponding to this stencil
  static constexpr int nbQuadPts = QUADRATURE_NUM_POINTS[stencilId];
  static constexpr int nbQuadPts3d = nbQuadPts * nbQuadPts;


}; // class ComputeFluxesFunctor

// =======================================================================
// =======================================================================
/**
 * Recompute MOOD fluxes around flagged cells.
 *
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 * - FluxData_z may or may not be allocated (depending dim==2 or 3).
 *
 */
template <int dim, int degree>
class RecomputeFluxesFunctor : public MoodBaseFunctor<dim, degree>
{

public:
  using typename MoodBaseFunctor<dim, degree>::DataArray;
  using typename MoodBaseFunctor<dim, degree>::HydroState;
  using MonomMap = typename mood::MonomialMap<dim, degree>::MonomMap;

  /**
   * Constructor for 2D/3D.
   */
  RecomputeFluxesFunctor(HydroParams params,
                         MonomMap    monomMap,
                         DataArray   Udata,
                         DataArray   Flags,
                         DataArray   FluxData_x,
                         DataArray   FluxData_y,
                         DataArray   FluxData_z,
                         real_t      dtdx,
                         real_t      dtdy,
                         real_t      dtdz)
    : MoodBaseFunctor<dim, degree>(params, monomMap)
    , Udata(Udata)
    , Flags(Flags)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z)
    , dtdx(dtdx)
    , dtdy(dtdy)
    , dtdz(dtdz){};

  ~RecomputeFluxesFunctor(){};

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

    // riemann solver states left/right
    HydroState UL, UR;

    // primitive variables left / right states
    HydroState qL, qR, qgdnv;
    real_t     c;

    // accumulate flux over all quadrature points
    HydroState flux;

    // current cell coordinates
    int i, j;
    index2coord(index, i, j, isize, jsize);

    // current flag (indicating if fluxes need to be recomputed)
    real_t flag = Flags(i, j, 0);
    real_t flagx = 0.0;
    real_t flagy = 0.0;

    if (i > 0)
      Flags(i - 1, j, 0);
    if (j > 0)
      Flags(i, j - 1, 0);

    if (flag > 0 or flagx > 0)
    {

      /*********************************
       * flux along DIR_X - FACE_XMIN
       *********************************/

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

      // for each variable,
      // retrieve UL / UR states
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // left  interface in neighbor cell
        UL[ivar] = Udata(i - 1, j, ivar);

        // right interface in current cell
        UR[ivar] = Udata(i, j, ivar);

      } // end for ivar

      // we can now perform the riemann solvers

      // convert to primitive variable before riemann solver
      this->computePrimitives(UL, &c, qL);
      this->computePrimitives(UR, &c, qR);

      // compute riemann flux
      //::ppkMHD::riemann_hydro(qL,qR,qgdnv,flux,this->params);
      ::ppkMHD::riemann_hll<HydroState2d>(qL, qR, qgdnv, flux, this->params);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_x(i, j, ivar) = flux[ivar] * dtdx;
    }

    if (flag > 0 or flagy > 0)
    {

      /*********************************
       * flux along DIR_Y - FACE_YMIN
       *********************************/

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

      // for each variable,
      // retrieve UL / UR states
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // left  interface in neighbor cell
        UL[ivar] = Udata(i, j - 1, ivar);

        // right interface in current cell
        UR[ivar] = Udata(i, j, ivar);

      } // end for ivar

      // we can now perform the riemann solvers

      // convert to primitive variable before riemann solver
      this->computePrimitives(UL, &c, qL);
      this->computePrimitives(UR, &c, qR);

      // compute riemann flux
      // swap IU and IV velocity
      this->swap(qL[IU], qL[IV]);
      this->swap(qR[IU], qR[IV]);

      // compute riemann flux
      //::ppkMHD::riemann_hydro(qL,qR,qgdnv,flux,this->params);
      ::ppkMHD::riemann_hll<HydroState2d>(qL, qR, qgdnv, flux, this->params);

      // swap again IU and IV
      this->swap(flux[IU], flux[IV]);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_y(i, j, ivar) = flux[ivar] * dtdy;

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

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    const real_t nbvar = this->params.nbvar;

    // riemann solver states left/right
    HydroState UL, UR;

    // primitive variables left / right states
    HydroState qL, qR, qgdnv;
    real_t     c;

    // accumulate flux over all quadrature points
    HydroState flux;

    // current cell coordinates
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // current flag (indicating if fluxes need to be recomputed)
    real_t flag = Flags(i, j, k, 0);
    real_t flagx = 0.0;
    real_t flagy = 0.0;
    real_t flagz = 0.0;

    if (i > 0)
      Flags(i - 1, j, k, 0);
    if (j > 0)
      Flags(i, j - 1, k, 0);
    if (k > 0)
      Flags(i, j, k - 1, 0);

    if (flag > 0 or flagx > 0)
    {

      /*********************************
       * flux along DIR_X - FACE_XMIN
       *********************************/

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

      // for each variable,
      // retrieve UL / UR states
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // left  interface in neighbor cell
        UL[ivar] = Udata(i - 1, j, k, ivar);

        // right interface in current cell
        UR[ivar] = Udata(i, j, k, ivar);

      } // end for ivar

      // we can now perform the riemann solvers

      // convert to primitive variable before riemann solver
      this->computePrimitives(UL, &c, qL);
      this->computePrimitives(UR, &c, qR);

      // compute riemann flux
      ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux, this->params);
      //::ppkMHD::riemann_hll<HydroState2d>(qL,qR,qgdnv,flux,this->params);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_x(i, j, k, ivar) = flux[ivar] * dtdx;

    } // end flux along X

    if (flag > 0 or flagy > 0)
    {

      /*********************************
       * flux along DIR_Y - FACE_YMIN
       *********************************/

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

      // for each variable,
      // retrieve UL / UR states
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // left  interface in neighbor cell
        UL[ivar] = Udata(i, j - 1, k, ivar);

        // right interface in current cell
        UR[ivar] = Udata(i, j, k, ivar);

      } // end for ivar

      // we can now perform the riemann solvers

      // convert to primitive variable before riemann solver
      this->computePrimitives(UL, &c, qL);
      this->computePrimitives(UR, &c, qR);

      // compute riemann flux
      // swap IU and IV velocity
      this->swap(qL[IU], qL[IV]);
      this->swap(qR[IU], qR[IV]);

      // compute riemann flux
      ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux, this->params);
      //::ppkMHD::riemann_hll<HydroState2d>(qL,qR,qgdnv,flux,this->params);

      // swap again IU and IV
      this->swap(flux[IU], flux[IV]);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_y(i, j, k, ivar) = flux[ivar] * dtdy;

    } // end flux along Y

    if (flag > 0 or flagz > 0)
    {

      /*********************************
       * flux along DIR_Z - FACE_ZMIN
       *********************************/

      // reset flux
      for (int ivar = 0; ivar < nbvar; ++ivar)
        flux[ivar] = 0.0;

      // for each variable,
      // retrieve UL / UR states
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // left  interface in neighbor cell
        UL[ivar] = Udata(i, j, k - 1, ivar);

        // right interface in current cell
        UR[ivar] = Udata(i, j, k, ivar);

      } // end for ivar

      // we can now perform the riemann solvers

      // convert to primitive variable before riemann solver
      this->computePrimitives(UL, &c, qL);
      this->computePrimitives(UR, &c, qR);

      // compute riemann flux
      // swap IU and IW velocity
      this->swap(qL[IU], qL[IW]);
      this->swap(qR[IU], qR[IW]);

      // compute riemann flux
      ::ppkMHD::riemann_hydro(qL, qR, qgdnv, flux, this->params);
      //::ppkMHD::riemann_hll<HydroState2d>(qL,qR,qgdnv,flux,this->params);

      // swap again IU and IW
      this->swap(flux[IU], flux[IW]);

      // finaly copy back the flux on device memory
      for (int ivar = 0; ivar < nbvar; ++ivar)
        FluxData_z(i, j, k, ivar) = flux[ivar] * dtdz;

    } // end flux along Z

  } // end functor 3d

  DataArray Udata;
  DataArray Flags;
  DataArray FluxData_x, FluxData_y, FluxData_z;
  real_t    dtdx, dtdy, dtdz;

}; // class RecomputeFluxesFunctor

} // namespace mood

#endif // MOOD_FLUXES_FUNCTORS_H_
