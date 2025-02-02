#ifndef SDM_BOUNDARIES_FUNCTORS_JET_H_
#define SDM_BOUNDARIES_FUNCTORS_JET_H_

#include "shared/HydroParams.h"        // for HydroParams
#include "shared/kokkos_shared.h"      // for Data arrays
#include "shared/problems/JetParams.h" // for Jet border condition

namespace ppkMHD
{
namespace sdm
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Functors to update ghost cells for the jet test case.
 *
 * reference:
 * "On positivity-preserving high order discontinuous Galerkin schemes for
 * compressible Euler equations on rectangular meshes", Xiangxiong Zhang,
 * Chi-Wang Shu, Journal of Computational Physics, Volume 229, Issue 23,
 * 20 November 2010, Pages 8918-8934
 * http://www.sciencedirect.com/science/article/pii/S0021999110004535
 *
 */
template <int dim, int N, FaceIdType faceId>
class MakeBoundariesFunctor_SDM_Jet : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;
  using typename SDMBaseFunctor<dim, N>::HydroState;

  static constexpr auto dofMap = DofMap<dim, N>;

  MakeBoundariesFunctor_SDM_Jet(HydroParams          params,
                                SDM_Geometry<dim, N> sdm_geom,
                                JetParams            jparams,
                                DataArray            Udata)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , jparams(jparams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams          params,
        SDM_Geometry<dim, N> sdm_geom,
        JetParams            jparams,
        DataArray            Udata,
        int                  nbIter)
  {
    MakeBoundariesFunctor_SDM_Jet<dim, N, faceId> functor(params, sdm_geom, jparams, Udata);
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
    // const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
#else
    // const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    // const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;

    // const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    // jet conservative variables
    const real_t rho1 = jparams.rho1;
    const real_t rho_u1 = jparams.rho_u1;
    const real_t rho_v1 = jparams.rho_v1;
    const real_t e_tot1 = jparams.e_tot1;

    // bulk conservative variables
    const real_t rho2 = jparams.rho2;
    const real_t rho_u2 = jparams.rho_u2;
    const real_t rho_v2 = jparams.rho_v2;
    const real_t e_tot2 = jparams.e_tot2;

    const real_t pos_jet = jparams.pos_jet;
    const real_t width_jet = jparams.width_jet;

    int i, j;

    // int boundary_type;

    int i0, j0;

    if (faceId == FACE_XMIN)
    {

      // boundary xmin (inflow / outflow)

      j = index / ghostWidth;
      i = index - j * ghostWidth;

      if (j >= jmin and j <= jmax and i >= 0 and i < ghostWidth)
      {

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            real_t y = ymin + (j + ny * j_mpi - ghostWidth) * dy;
            y += this->sdm_geom.solution_pts_1d(idy) * dy;

            if (y > pos_jet - 0.5 * width_jet and y < pos_jet + 0.5 * width_jet)
            { // jet / inflow

              Udata(i, j, dofMap(idx, idy, 0, ID)) = rho1;
              Udata(i, j, dofMap(idx, idy, 0, IE)) = e_tot1;
              Udata(i, j, dofMap(idx, idy, 0, IU)) = rho_u1;
              Udata(i, j, dofMap(idx, idy, 0, IV)) = rho_v1;
            }
            else
            { // bulk

              Udata(i, j, dofMap(idx, idy, 0, ID)) = rho2;
              Udata(i, j, dofMap(idx, idy, 0, IE)) = e_tot2;
              Udata(i, j, dofMap(idx, idy, 0, IU)) = rho_u2;
              Udata(i, j, dofMap(idx, idy, 0, IV)) = rho_v2;
            }

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

      if (j >= jmin and j <= jmax and i >= nx + ghostWidth and i <= nx + 2 * ghostWidth - 1)
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

      // boundary ymin : outflow

      i = index / ghostWidth;
      j = index - i * ghostWidth;

      if (i >= imin and i <= imax and j >= 0 and j < ghostWidth)
      {

        j0 = ghostWidth;

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            for (int iVar = 0; iVar < nbvar; iVar++)
            {
              Udata(i, j, dofMap(idx, idy, 0, iVar)) =
                Udata(i, j0, dofMap(idx, /*N-1-*/ idy, 0, iVar));
            }

          } // end for idx
        } // end for idy

      } // end if i,j

    } // end FACE_YMIN

    if (faceId == FACE_YMAX)
    {

      // boundary ymax : outflow

      i = index / ghostWidth;
      j = index - i * ghostWidth;
      j += (ny + ghostWidth);

      if (i >= imin and i <= imax and j >= ny + ghostWidth and j <= ny + 2 * ghostWidth - 1)
      {

        j0 = ny + ghostWidth - 1;

        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            for (int iVar = 0; iVar < nbvar; iVar++)
            {
              Udata(i, j, dofMap(idx, idy, 0, iVar)) =
                Udata(i, j0, dofMap(idx, /*N-1-*/ idy, 0, iVar));
            }

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

    /* PARTIALLY UNIMPLEMENTED */

    const int nx = this->params.nx;
    const int ny = this->params.ny;
    const int nz = this->params.nz;

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    // const int ksize = this->params.ksize;

    const int ghostWidth = this->params.ghostWidth;
    const int nbvar = this->params.nbvar;

    const int imin = this->params.imin;
    const int imax = this->params.imax;

    const int jmin = this->params.jmin;
    const int jmax = this->params.jmax;

    const int kmin = this->params.kmin;
    const int kmax = this->params.kmax;

#ifdef USE_MPI
    // const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
    const int k_mpi = this->params.myMpiPos[IZ];
#else
    // const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    // const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;

    // const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    // jet conservative variables
    const real_t rho1 = jparams.rho1;
    const real_t rho_u1 = jparams.rho_u1;
    const real_t rho_v1 = jparams.rho_v1;
    const real_t rho_w1 = jparams.rho_w1;
    const real_t e_tot1 = jparams.e_tot1;

    // bulk conservative variables
    const real_t rho2 = jparams.rho2;
    const real_t rho_u2 = jparams.rho_u2;
    const real_t rho_v2 = jparams.rho_v2;
    const real_t rho_w2 = jparams.rho_w2;
    const real_t e_tot2 = jparams.e_tot2;

    const real_t pos_jet = jparams.pos_jet;
    const real_t width_jet = jparams.width_jet;

    int i, j, k;

    // int boundary_type;

    int i0, j0, k0;

    if (faceId == FACE_XMIN)
    {

      // boundary xmin (inflow / outflow)

      // boundary xmin (index = i + j * ghostWidth + k * ghostWidth*jsize)
      k = index / (ghostWidth * jsize);
      j = (index - k * ghostWidth * jsize) / ghostWidth;
      i = index - j * ghostWidth - k * ghostWidth * jsize;

      if (k >= kmin and k <= kmax and j >= jmin and j <= jmax and i >= 0 and i < ghostWidth)
      {

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idy = 0; idy < N; ++idy)
          {
            for (int idx = 0; idx < N; ++idx)
            {

              real_t y = ymin + (j + ny * j_mpi - ghostWidth) * dy;
              real_t z = zmin + (k + nz * k_mpi - ghostWidth) * dz;

              y += this->sdm_geom.solution_pts_1d(idy) * dy;
              z += this->sdm_geom.solution_pts_1d(idz) * dz;

              real_t radius2 = (y - pos_jet) * (y - pos_jet) + (z - pos_jet) * (z - pos_jet);

              if (radius2 < 0.25 * width_jet * width_jet)
              { // jet / inflow

                Udata(i, j, k, dofMap(idx, idy, idz, ID)) = rho1;
                Udata(i, j, k, dofMap(idx, idy, idz, IE)) = e_tot1;
                Udata(i, j, k, dofMap(idx, idy, idz, IU)) = rho_u1;
                Udata(i, j, k, dofMap(idx, idy, idz, IV)) = rho_v1;
                Udata(i, j, k, dofMap(idx, idy, idz, IW)) = rho_w1;
              }
              else
              { // bulk

                Udata(i, j, k, dofMap(idx, idy, idz, ID)) = rho2;
                Udata(i, j, k, dofMap(idx, idy, idz, IE)) = e_tot2;
                Udata(i, j, k, dofMap(idx, idy, idz, IU)) = rho_u2;
                Udata(i, j, k, dofMap(idx, idy, idz, IV)) = rho_v2;
                Udata(i, j, k, dofMap(idx, idy, idz, IW)) = rho_w2;
              }

            } // end idx
          } // end idy
        } // end idz

      } // end if inside border

    } // end FACE_XMIN

    if (faceId == FACE_XMAX)
    {

      // boundary xmax (outflow)
      // boundary xmax (index = i + j *ghostWidth + k * ghostWidth*jsize)
      // same i,j,k as xmin, except translation along x-axis
      k = index / (ghostWidth * jsize);
      j = (index - k * ghostWidth * jsize) / ghostWidth;
      i = index - j * ghostWidth - k * ghostWidth * jsize;

      i += (nx + ghostWidth);

      if (k >= kmin and k <= kmax and j >= jmin and j <= jmax and i >= nx + ghostWidth and
          i <= nx + 2 * ghostWidth - 1)
      {

        i0 = nx + ghostWidth - 1;

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idy = 0; idy < N; ++idy)
          {
            for (int idx = 0; idx < N; ++idx)
            {
              for (int iVar = 0; iVar < nbvar; iVar++)
              {
                // copy Dof from cell i0,j,k into cell i,j,k with a mirror
                Udata(i, j, k, dofMap(idx, idy, idz, iVar)) =
                  Udata(i0, j, k, dofMap(idx, idy, idz, iVar));
              }
            } // end for idx
          } // end for idy
        } // end for idz

      } // end if inside border

    } // end FACE_XMAX

    if (faceId == FACE_YMIN)
    {

      // boundary ymin : outflow

      k = index / (isize * ghostWidth);
      j = (index - k * isize * ghostWidth) / isize;
      i = index - j * isize - k * isize * ghostWidth;

      if (k >= kmin and k <= kmax and j >= 0 and j < ghostWidth and i >= imin and i <= imax)
      {

        j0 = ghostWidth;

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idy = 0; idy < N; ++idy)
          {
            for (int idx = 0; idx < N; ++idx)
            {

              for (int iVar = 0; iVar < nbvar; iVar++)
              {
                Udata(i, j, k, dofMap(idx, idy, idz, iVar)) =
                  Udata(i, j0, k, dofMap(idx, idy, idz, iVar));
              }

            } // end for idx
          } // end for idy
        } // end for idz

      } // end if inside border

    } // end FACE_YMIN

    if (faceId == FACE_YMAX)
    {

      // boundary ymax : outflow

      k = index / (isize * ghostWidth);
      j = (index - k * isize * ghostWidth) / isize;
      i = index - j * isize - k * isize * ghostWidth;

      j += (ny + ghostWidth);

      if (k >= kmin and k <= kmax and j >= ny + ghostWidth and j <= ny + 2 * ghostWidth - 1 and
          i >= imin and i <= imax)
      {

        j0 = ny + ghostWidth - 1;

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idy = 0; idy < N; ++idy)
          {
            for (int idx = 0; idx < N; ++idx)
            {

              for (int iVar = 0; iVar < nbvar; iVar++)
              {
                Udata(i, j, k, dofMap(idx, idy, idz, iVar)) =
                  Udata(i, j0, k, dofMap(idx, idy, idz, iVar));
              }

            } // end idx
          } // end idy
        } // end idz

      } // end if inside border

    } // end FACE_YMAX

    if (faceId == FACE_ZMIN)
    {

      // boundary zmin : outflow

      k = index / (isize * jsize);
      j = (index - k * isize * jsize) / isize;
      i = index - j * isize - k * isize * jsize;

      if (k >= 0 and k < ghostWidth and j >= jmin and j <= jmax and i >= imin and i <= imax)
      {

        k0 = ghostWidth;

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idy = 0; idy < N; ++idy)
          {
            for (int idx = 0; idx < N; ++idx)
            {

              for (int iVar = 0; iVar < nbvar; iVar++)
              {
                Udata(i, j, k, dofMap(idx, idy, idz, iVar)) =
                  Udata(i, j, k0, dofMap(idx, idy, idz, iVar));
              }

            } // end for idx
          } // end for idy
        } // end for idz

      } // end if inside border

    } // end FACE_ZMIN

    if (faceId == FACE_ZMAX)
    {

      // boundary zmax : outflow

      k = index / (isize * jsize);
      j = (index - k * isize * jsize) / isize;
      i = index - j * isize - k * isize * jsize;

      k += (nz + ghostWidth);

      if (k >= nz + ghostWidth and k <= nz + 2 * ghostWidth - 1 and j >= jmin and j <= jmax and
          i >= imin and i <= imax)
      {

        k0 = nz + ghostWidth - 1;

        for (int idz = 0; idz < N; ++idz)
        {
          for (int idy = 0; idy < N; ++idy)
          {
            for (int idx = 0; idx < N; ++idx)
            {

              for (int iVar = 0; iVar < nbvar; iVar++)
              {
                Udata(i, j, k, dofMap(idx, idy, idz, iVar)) =
                  Udata(i, j, k0, dofMap(idx, idy, idz, iVar));
              }

            } // end idx
          } // end idy
        } // end idz

      } // end if inside border

    } // end FACE_ZMAX

  } // operator () - 3d

  JetParams jparams;
  DataArray Udata;

}; // MakeBoundariesFunctor_SDM_Jet

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_BOUNDARIES_FUNCTORS_JET_H_
