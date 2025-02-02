#ifndef SDM_BOUNDARIES_FUNCTORS_WEDGE_H_
#define SDM_BOUNDARIES_FUNCTORS_WEDGE_H_

#include "shared/HydroParams.h"          // for HydroParams
#include "shared/kokkos_shared.h"        // for Data arrays
#include "shared/problems/WedgeParams.h" // for Wedge border condition

namespace ppkMHD
{
namespace sdm
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Functors to update ghost cells (2D only) for the test case wedge,
 * also called Double Mach reflection.
 *
 * See http://amroc.sourceforge.net/examples/euler/2d/html/ramp_n.htm
 *
 * This border condition is time-dependent.
 */
template <int dim, int N, FaceIdType faceId>
class MakeBoundariesFunctor_SDM_Wedge : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;
  using typename SDMBaseFunctor<dim, N>::HydroState;

  static constexpr auto dofMap = DofMap<dim, N>;

  MakeBoundariesFunctor_SDM_Wedge(HydroParams          params,
                                  SDM_Geometry<dim, N> sdm_geom,
                                  WedgeParams          wparams,
                                  DataArray            Udata)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , wparams(wparams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams          params,
        SDM_Geometry<dim, N> sdm_geom,
        WedgeParams          wparams,
        DataArray            Udata,
        int                  nbIter)
  {
    MakeBoundariesFunctor_SDM_Wedge<dim, N, faceId> functor(params, sdm_geom, wparams, Udata);
    Kokkos::parallel_for(nbIter, functor);
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

    const int nx = this->params.nx;
    const int ny = this->params.ny;

    const int ghostWidth = this->params.ghostWidth;
    const int nbvar = this->params.nbvar;

    const int imin = this->params.imin;
    const int imax = this->params.imax;

    const int jmin = this->params.jmin;
    const int jmax = this->params.jmax;

#ifdef USE_MPI
    const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    const real_t rho1 = wparams.rho1;
    const real_t rho_u1 = wparams.rho_u1;
    const real_t rho_v1 = wparams.rho_v1;
    const real_t e_tot1 = wparams.e_tot1;

    const real_t rho2 = wparams.rho2;
    const real_t rho_u2 = wparams.rho_u2;
    const real_t rho_v2 = wparams.rho_v2;
    const real_t e_tot2 = wparams.e_tot2;

    int i, j;

    // int boundary_type;

    int i0, j0;

    if (faceId == FACE_XMIN)
    {

      // boundary xmin (inflow)

      j = index / ghostWidth;
      i = index - j * ghostWidth;

      if (j >= jmin && j <= jmax && i >= 0 && i < ghostWidth)
      {

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {
            Udata(i, j, dofMap(idx, idy, 0, ID)) = rho1;
            Udata(i, j, dofMap(idx, idy, 0, IE)) = e_tot1;
            Udata(i, j, dofMap(idx, idy, 0, IU)) = rho_u1;
            Udata(i, j, dofMap(idx, idy, 0, IV)) = rho_v1;
          } // end idx
        } // end idy
      }

    } // end FACE_XMIN

    if (faceId == FACE_XMAX)
    {

      // boundary xmax (outflow)
      j = index / ghostWidth;
      i = index - j * ghostWidth;
      i += (nx + ghostWidth);

      if (j >= jmin && j <= jmax && i >= nx + ghostWidth && i <= nx + 2 * ghostWidth - 1)
      {

        i0 = nx + ghostWidth - 1;

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {
            for (int iVar = 0; iVar < nbvar; iVar++)
            {
              // copy Dof from cell i0,j into cell i,j with a mirror
              Udata(i, j, dofMap(idx, idy, 0, iVar)) =
                Udata(i0, j, dofMap(N - 1 - idx, idy, 0, iVar));
            }
          } // end for idx
        } // end for idy
      }

    } // end FACE_XMAX

    if (faceId == FACE_YMIN)
    {

      // boundary ymin
      // if (x <  x_f) inflow
      // else          reflective

      i = index / ghostWidth;
      j = index - i * ghostWidth;

      if (i >= imin && i <= imax && j >= 0 && j < ghostWidth)
      {

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            // lower left corner
            real_t x = xmin + (i + nx * i_mpi - ghostWidth) * dx;
            x += this->sdm_geom.solution_pts_1d(idx) * dx;

            if (x < wparams.x_f)
            { // inflow

              Udata(i, j, dofMap(idx, idy, 0, ID)) = rho1;
              Udata(i, j, dofMap(idx, idy, 0, IE)) = e_tot1;
              Udata(i, j, dofMap(idx, idy, 0, IU)) = rho_u1;
              Udata(i, j, dofMap(idx, idy, 0, IV)) = rho_v1;
            }
            else
            { // reflective

              // mirror DoFs idy <-> N-1-idy

              real_t sign = 1.0;
              j0 = 2 * ghostWidth - 1 - j;

              for (int iVar = 0; iVar < nbvar; iVar++)
              {
                if (iVar == IV)
                  sign = -ONE_F;
                Udata(i, j, dofMap(idx, idy, 0, iVar)) =
                  Udata(i, j0, dofMap(idx, N - 1 - idy, 0, iVar)) * sign;
              }

            } // end inflow / reflective

          } // end for idx
        } // end for idy

      } // end if i,j

    } // end FACE_YMIN

    if (faceId == FACE_YMAX)
    {

      // boundary ymax
      // if (x <  x_f + y/slope_f + delta_x) inflow
      // else                                outflow

      i = index / ghostWidth;
      j = index - i * ghostWidth;
      j += (ny + ghostWidth);

      if (i >= imin && i <= imax && j >= ny + ghostWidth && j <= ny + 2 * ghostWidth - 1)
      {

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            // lower left corner
            real_t x = xmin + (i + nx * i_mpi - ghostWidth) * dx;
            real_t y = ymin + (j + ny * j_mpi - ghostWidth) * dy;

            x += this->sdm_geom.solution_pts_1d(idx) * dx;
            y += this->sdm_geom.solution_pts_1d(idy) * dy;

            if (x < wparams.x_f + y / wparams.slope_f + wparams.delta_x)
            { // inflow

              Udata(i, j, dofMap(idx, idy, 0, ID)) = rho1;
              Udata(i, j, dofMap(idx, idy, 0, IP)) = e_tot1;
              Udata(i, j, dofMap(idx, idy, 0, IU)) = rho_u1;
              Udata(i, j, dofMap(idx, idy, 0, IV)) = rho_v1;
            }
            else
            { // outflow

              // j0=ny+ghostWidth-1;

              // // copy the last Dof from cell i,j0 into every Dof of cell i,j
              // for ( int iVar=0; iVar<nbvar; iVar++ ) {
              // 	Udata(i,j,dofMap(idx,idy,0,iVar)) =
              // 	  Udata(i,j0,dofMap(idx,N-1-idy,0,iVar));
              // }

              Udata(i, j, dofMap(idx, idy, 0, ID)) = rho2;
              Udata(i, j, dofMap(idx, idy, 0, IP)) = e_tot2;
              Udata(i, j, dofMap(idx, idy, 0, IU)) = rho_u2;
              Udata(i, j, dofMap(idx, idy, 0, IV)) = rho_v2;

            } // end inflow / outflow

          } // end idx
        } // end idy

      } // end if i,j

    } // end FACE_YMAX

  } // operator () - 2d

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

    /* UNIMPLEMENTED */

  } // operator () - 3d

  WedgeParams wparams;
  DataArray   Udata;

}; // MakeBoundariesFunctor_SDM_Wedge

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_BOUNDARIES_FUNCTORS_WEDGE_H_
