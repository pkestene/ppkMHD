#ifndef SDM_FLUX_FUNCTORS_H_
#define SDM_FLUX_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/RiemannSolvers.h"
#include "shared/EulerEquations.h"

namespace ppkMHD
{
namespace sdm
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor computes fluxes at fluxes points taking
 * as input conservative variables at fluxes points.
 *
 * We first compute fluxes at flux points interior to cell, then
 * handle the end points.
 */
template <int dim, int N, int dir>
class ComputeFluxAtFluxPoints_Functor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;
  using typename SDMBaseFunctor<dim, N>::HydroState;

  static constexpr auto dofMapF = DofMapFlux<dim, N, dir>;

  ComputeFluxAtFluxPoints_Functor(HydroParams                 params,
                                  SDM_Geometry<dim, N>        sdm_geom,
                                  ppkMHD::EulerEquations<dim> euler,
                                  DataArray                   UdataFlux)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , euler(euler)
    , UdataFlux(UdataFlux){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams                 params,
        SDM_Geometry<dim, N>        sdm_geom,
        ppkMHD::EulerEquations<dim> euler,
        DataArray                   UdataFlux)
  {
    int64_t nbCells =
      (dim == 2) ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    ComputeFluxAtFluxPoints_Functor functor(params, sdm_geom, euler, UdataFlux);
    Kokkos::parallel_for("ComputeFluxAtFluxPoints_Functor", nbCells, functor);
  }

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    // state variable for conservative variables, and flux
    HydroState q = {}, flux;

    /*
     * first handle interior point, then end points.
     */

    // =========================
    // ========= DIR X =========
    // =========================
    // loop over cell DoF's
    if (dir == IX)
    {

      for (int idy = 0; idy < N; ++idy)
      {

        // interior points along direction X
        for (int idx = 1; idx < N; ++idx)
        {

          // retrieve state conservative variables
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {

            q[ivar] = UdataFlux(i, j, dofMapF(idx, idy, 0, ivar));
          }

          // compute pressure
          real_t p = euler.compute_pressure(q, this->params.settings.gamma0);

          // compute flux along X direction
          euler.flux_x(q, p, flux);

          // copy back interpolated value
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {

            UdataFlux(i, j, dofMapF(idx, idy, 0, ivar)) = flux[ivar];

          } // end for ivar

        } // end for idx

      } // end for idy

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (i > 0 and i < isize)
      {

        for (int idy = 0; idy < N; ++idy)
        {

          // conservative state
          HydroState qL = {}, qR = {};

          // primitive state
          HydroState wL, wR;

          HydroState qgdnv;

          // when idx == 0, get right and left state
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {
            qL[ivar] = UdataFlux(i - 1, j, dofMapF(N, idy, 0, ivar));
            qR[ivar] = UdataFlux(i, j, dofMapF(0, idy, 0, ivar));
          }

          // convert to primitive
          euler.convert_to_primitive(qR, wR, this->params.settings.gamma0);
          euler.convert_to_primitive(qL, wL, this->params.settings.gamma0);

          // riemann solver
          ppkMHD::riemann_hydro(wL, wR, qgdnv, flux, this->params);

          // copy back result in current cell and in neighbor
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {
            UdataFlux(i - 1, j, dofMapF(N, idy, 0, ivar)) = flux[ivar];
            UdataFlux(i, j, dofMapF(0, idy, 0, ivar)) = flux[ivar];
          }

        } // end for idy

      } // end safe-guard

    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    // loop over cell DoF's
    if (dir == IY)
    {

      for (int idx = 0; idx < N; ++idx)
      {

        // interior points along direction X
        for (int idy = 1; idy < N; ++idy)
        {

          // for each variables
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {

            q[ivar] = UdataFlux(i, j, dofMapF(idx, idy, 0, ivar));
          }

          // compute pressure
          real_t p = euler.compute_pressure(q, this->params.settings.gamma0);

          // compute flux along Y direction
          euler.flux_y(q, p, flux);

          // copy back interpolated value
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {

            UdataFlux(i, j, dofMapF(idx, idy, 0, ivar)) = flux[ivar];

          } // end for ivar

        } // end for idy

      } // end for idx

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (j > 0 and j < jsize)
      {

        for (int idx = 0; idx < N; ++idx)
        {

          // conservative state
          HydroState qL = {}, qR = {};

          // primitive state
          HydroState wL, wR;

          HydroState qgdnv;

          // when idy == 0, get right and left state
          for (int ivar = 0; ivar < nbvar; ++ivar)
          {
            qL[ivar] = UdataFlux(i, j - 1, dofMapF(idx, N, 0, ivar));
            qR[ivar] = UdataFlux(i, j, dofMapF(idx, 0, 0, ivar));
          }

          // convert to primitive : q -> w
          euler.convert_to_primitive(qR, wR, this->params.settings.gamma0);
          euler.convert_to_primitive(qL, wL, this->params.settings.gamma0);

          // riemann solver
          this->swap(wL[IU], wL[IV]);
          this->swap(wR[IU], wR[IV]);
          ppkMHD::riemann_hydro(wL, wR, qgdnv, flux, this->params);

          // copy back results in current cell as well as in neighbor
          UdataFlux(i, j, dofMapF(idx, 0, 0, ID)) = flux[ID];
          UdataFlux(i, j, dofMapF(idx, 0, 0, IE)) = flux[IE];
          UdataFlux(i, j, dofMapF(idx, 0, 0, IU)) = flux[IV]; // swap again
          UdataFlux(i, j, dofMapF(idx, 0, 0, IV)) = flux[IU]; // swap again

          UdataFlux(i, j - 1, dofMapF(idx, N, 0, ID)) = flux[ID];
          UdataFlux(i, j - 1, dofMapF(idx, N, 0, IE)) = flux[IE];
          UdataFlux(i, j - 1, dofMapF(idx, N, 0, IU)) = flux[IV]; // swap again
          UdataFlux(i, j - 1, dofMapF(idx, N, 0, IV)) = flux[IU]; // swap again

        } // end for idx

      } // end safe-guard

    } // end for dir IY

  } // 2d


  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // state variable for conservative variables, and flux
    HydroState q = {}, flux;

    /*
     * first handle interior point, then end points.
     */

    // =========================
    // ========= DIR X =========
    // =========================
    if (dir == IX)
    {

      for (int idz = 0; idz < N; ++idz)
      {
        for (int idy = 0; idy < N; ++idy)
        {

          // interior points along direction X
          for (int idx = 1; idx < N; ++idx)
          {

            // retrieve state conservative variables
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {

              q[ivar] = UdataFlux(i, j, k, dofMapF(idx, idy, idz, ivar));
            }

            // compute pressure
            real_t p = euler.compute_pressure(q, this->params.settings.gamma0);

            // compute flux along X direction
            euler.flux_x(q, p, flux);

            // copy back interpolated value
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {

              UdataFlux(i, j, k, dofMapF(idx, idy, idz, ivar)) = flux[ivar];

            } // end for ivar

          } // end for idx

        } // end for idy
      } // end for idz

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (i > 0 and i < isize)
      {

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idy = 0; idy < N; ++idy)
          {

            // conservative state
            HydroState qL = {}, qR = {};

            // primitive state
            HydroState wL, wR;

            HydroState qgdnv;

            // when idx == 0, get right and left state
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {
              qL[ivar] = UdataFlux(i - 1, j, k, dofMapF(N, idy, idz, ivar));
              qR[ivar] = UdataFlux(i, j, k, dofMapF(0, idy, idz, ivar));
            }

            // convert to primitive
            euler.convert_to_primitive(qR, wR, this->params.settings.gamma0);
            euler.convert_to_primitive(qL, wL, this->params.settings.gamma0);

            // riemann solver
            ppkMHD::riemann_hydro(wL, wR, qgdnv, flux, this->params);

            // copy flux
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {
              UdataFlux(i - 1, j, k, dofMapF(N, idy, idz, ivar)) = flux[ivar];
              UdataFlux(i, j, k, dofMapF(0, idy, idz, ivar)) = flux[ivar];
            }

          } // end for idy
        } // end for idz

      } // end safe-guard

    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    if (dir == IY)
    {

      for (int idz = 0; idz < N; ++idz)
      {
        for (int idx = 0; idx < N; ++idx)
        {

          // interior points along direction Y
          for (int idy = 1; idy < N; ++idy)
          {

            // retrieve state conservative variables
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {

              q[ivar] = UdataFlux(i, j, k, dofMapF(idx, idy, idz, ivar));
            }

            // compute pressure
            real_t p = euler.compute_pressure(q, this->params.settings.gamma0);

            // compute flux along Y direction
            euler.flux_y(q, p, flux);

            // copy back interpolated value
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {

              UdataFlux(i, j, k, dofMapF(idx, idy, idz, ivar)) = flux[ivar];

            } // end for ivar

          } // end for idy

        } // end for idx
      } // end for idz

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (j > 0 and j < jsize)
      {

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            // conservative state
            HydroState qL = {}, qR = {};

            // primitive state
            HydroState wL, wR;

            HydroState qgdnv;

            // when idy == 0, get right and left state
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {
              qL[ivar] = UdataFlux(i, j - 1, k, dofMapF(idx, N, idz, ivar));
              qR[ivar] = UdataFlux(i, j, k, dofMapF(idx, 0, idz, ivar));
            }

            // convert to primitive
            euler.convert_to_primitive(qR, wR, this->params.settings.gamma0);
            euler.convert_to_primitive(qL, wL, this->params.settings.gamma0);

            // riemann solver
            this->swap(wL[IU], wL[IV]);
            this->swap(wR[IU], wR[IV]);
            ppkMHD::riemann_hydro(wL, wR, qgdnv, flux, this->params);

            // copy back results in current and neighbor cells
            UdataFlux(i, j - 1, k, dofMapF(idx, N, idz, ID)) = flux[ID];
            UdataFlux(i, j - 1, k, dofMapF(idx, N, idz, IE)) = flux[IE];
            UdataFlux(i, j - 1, k, dofMapF(idx, N, idz, IU)) = flux[IV]; // swap again
            UdataFlux(i, j - 1, k, dofMapF(idx, N, idz, IV)) = flux[IU]; // swap again
            UdataFlux(i, j - 1, k, dofMapF(idx, N, idz, IW)) = flux[IW];

            UdataFlux(i, j, k, dofMapF(idx, 0, idz, ID)) = flux[ID];
            UdataFlux(i, j, k, dofMapF(idx, 0, idz, IE)) = flux[IE];
            UdataFlux(i, j, k, dofMapF(idx, 0, idz, IU)) = flux[IV]; // swap again
            UdataFlux(i, j, k, dofMapF(idx, 0, idz, IV)) = flux[IU]; // swap again
            UdataFlux(i, j, k, dofMapF(idx, 0, idz, IW)) = flux[IW];

          } // end for idx
        } // end for idz

      } // end safe-guard

    } // end for dir IY

    // =========================
    // ========= DIR Z =========
    // =========================
    if (dir == IZ)
    {

      for (int idy = 0; idy < N; ++idy)
      {
        for (int idx = 0; idx < N; ++idx)
        {

          // interior points along direction Z
          for (int idz = 1; idz < N; ++idz)
          {

            // retrieve state conservative variables
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {

              q[ivar] = UdataFlux(i, j, k, dofMapF(idx, idy, idz, ivar));
            }

            // compute pressure
            real_t p = euler.compute_pressure(q, this->params.settings.gamma0);

            // compute flux along Z direction
            euler.flux_z(q, p, flux);

            // copy back interpolated value
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {

              UdataFlux(i, j, k, dofMapF(idx, idy, idz, ivar)) = flux[ivar];

            } // end for ivar

          } // end for idz

        } // end for idx
      } // end for idy

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (k > 0 and k < ksize)
      {

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            // conservative state
            HydroState qL = {}, qR = {};

            // primitive state
            HydroState wL, wR;

            HydroState qgdnv;

            // when idz == 0, get right and left state
            for (int ivar = 0; ivar < nbvar; ++ivar)
            {
              qL[ivar] = UdataFlux(i, j, k - 1, dofMapF(idx, idy, N, ivar));
              qR[ivar] = UdataFlux(i, j, k, dofMapF(idx, idy, 0, ivar));
            }

            // convert to primitive
            euler.convert_to_primitive(qR, wR, this->params.settings.gamma0);
            euler.convert_to_primitive(qL, wL, this->params.settings.gamma0);

            // riemann solver
            this->swap(wL[IU], wL[IW]);
            this->swap(wR[IU], wR[IW]);
            ppkMHD::riemann_hydro(wL, wR, qgdnv, flux, this->params);

            // copy back results in current and neighbor cells
            UdataFlux(i, j, k - 1, dofMapF(idx, idy, N, ID)) = flux[ID];
            UdataFlux(i, j, k - 1, dofMapF(idx, idy, N, IE)) = flux[IE];
            UdataFlux(i, j, k - 1, dofMapF(idx, idy, N, IU)) = flux[IW]; // swap again
            UdataFlux(i, j, k - 1, dofMapF(idx, idy, N, IV)) = flux[IV];
            UdataFlux(i, j, k - 1, dofMapF(idx, idy, N, IW)) = flux[IU]; // swap again

            UdataFlux(i, j, k, dofMapF(idx, idy, 0, ID)) = flux[ID];
            UdataFlux(i, j, k, dofMapF(idx, idy, 0, IE)) = flux[IE];
            UdataFlux(i, j, k, dofMapF(idx, idy, 0, IU)) = flux[IW]; // swap again
            UdataFlux(i, j, k, dofMapF(idx, idy, 0, IV)) = flux[IV];
            UdataFlux(i, j, k, dofMapF(idx, idy, 0, IW)) = flux[IU]; // swap again

          } // end for idx
        } // end for idy

      } // end safe-guard

    } // end for dir IZ

  } // 3d

  ppkMHD::EulerEquations<dim> euler;
  DataArray                   UdataFlux;

}; // class ComputeFluxAtFluxPoints_Functor

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_FLUX_FUNCTORS_H_
