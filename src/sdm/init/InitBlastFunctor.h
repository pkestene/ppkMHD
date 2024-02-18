#ifndef SDM_INIT_BLAST_FUNCTOR_H_
#define SDM_INIT_BLAST_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/problems/BlastParams.h"

namespace ppkMHD
{
namespace sdm
{

/*************************************************/
/*************************************************/
/*************************************************/
template <int dim, int N>
class InitBlastFunctor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;

  static constexpr auto dofMap = DofMap<dim, N>;

  InitBlastFunctor(HydroParams          params,
                   SDM_Geometry<dim, N> sdm_geom,
                   BlastParams          bParams,
                   DataArray            Udata)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , bParams(bParams)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, SDM_Geometry<dim, N> sdm_geom, BlastParams bparams, DataArray Udata)
  {
    int nbCells =
      dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    InitBlastFunctor functor(params, sdm_geom, bparams, Udata);
    Kokkos::parallel_for("InitBlastFunctor", nbCells, functor);
  }

  /*
   * 2D version.
   */
  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const int nx = this->params.nx;
    const int ny = this->params.ny;

    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    const real_t gamma0 = this->params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius = bParams.blast_radius;
    const real_t radius2 = blast_radius * blast_radius;
    const real_t blast_center_x = bParams.blast_center_x;
    const real_t blast_center_y = bParams.blast_center_y;
    const real_t blast_density_in = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out = bParams.blast_pressure_out;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    // loop over cell DoF's
    for (int idy = 0; idy < N; ++idy)
    {
      for (int idx = 0; idx < N; ++idx)
      {

        // lower left corner
        real_t x = xmin + (i + nx * i_mpi - ghostWidth) * dx;
        real_t y = ymin + (j + ny * j_mpi - ghostWidth) * dy;

        // DoF location
        x += this->sdm_geom.solution_pts_1d(idx) * dx;
        y += this->sdm_geom.solution_pts_1d(idy) * dy;

        real_t d2 =
          (x - blast_center_x) * (x - blast_center_x) + (y - blast_center_y) * (y - blast_center_y);

        if (d2 < radius2)
        {
          Udata(i, j, dofMap(idx, idy, 0, ID)) = blast_density_in;
          Udata(i, j, dofMap(idx, idy, 0, IE)) = blast_pressure_in / (gamma0 - 1.0);
          Udata(i, j, dofMap(idx, idy, 0, IU)) = 0.0;
          Udata(i, j, dofMap(idx, idy, 0, IV)) = 0.0;
        }
        else
        {
          Udata(i, j, dofMap(idx, idy, 0, ID)) = blast_density_out;
          Udata(i, j, dofMap(idx, idy, 0, IE)) = blast_pressure_out / (gamma0 - 1.0);
          Udata(i, j, dofMap(idx, idy, 0, IU)) = 0.0;
          Udata(i, j, dofMap(idx, idy, 0, IV)) = 0.0;
        }

      } // end for idx
    }   // end for idy

  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
    const int k_mpi = this->params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    const int nx = this->params.nx;
    const int ny = this->params.ny;
    const int nz = this->params.nz;

    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    const real_t gamma0 = this->params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius = bParams.blast_radius;
    const real_t radius2 = blast_radius * blast_radius;
    const real_t blast_center_x = bParams.blast_center_x;
    const real_t blast_center_y = bParams.blast_center_y;
    const real_t blast_center_z = bParams.blast_center_z;
    const real_t blast_density_in = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out = bParams.blast_pressure_out;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // loop over cell DoF's
    for (int idz = 0; idz < N; ++idz)
    {
      for (int idy = 0; idy < N; ++idy)
      {
        for (int idx = 0; idx < N; ++idx)
        {

          // lower left corner
          real_t x = xmin + (i + nx * i_mpi - ghostWidth) * dx;
          real_t y = ymin + (j + ny * j_mpi - ghostWidth) * dy;
          real_t z = zmin + (k + nz * k_mpi - ghostWidth) * dz;

          x += this->sdm_geom.solution_pts_1d(idx) * dx;
          y += this->sdm_geom.solution_pts_1d(idy) * dy;
          z += this->sdm_geom.solution_pts_1d(idz) * dz;

          real_t d2 = (x - blast_center_x) * (x - blast_center_x) +
                      (y - blast_center_y) * (y - blast_center_y) +
                      (z - blast_center_z) * (z - blast_center_z);

          if (d2 < radius2)
          {

            Udata(i, j, k, dofMap(idx, idy, idz, ID)) = blast_density_in;
            Udata(i, j, k, dofMap(idx, idy, idz, IE)) = blast_pressure_in / (gamma0 - 1.0);
            Udata(i, j, k, dofMap(idx, idy, idz, IU)) = 0.0;
            Udata(i, j, k, dofMap(idx, idy, idz, IV)) = 0.0;
            Udata(i, j, k, dofMap(idx, idy, idz, IW)) = 0.0;
          }
          else
          {

            Udata(i, j, k, dofMap(idx, idy, idz, ID)) = blast_density_out;
            Udata(i, j, k, dofMap(idx, idy, idz, IE)) = blast_pressure_out / (gamma0 - 1.0);
            Udata(i, j, k, dofMap(idx, idy, idz, IU)) = 0.0;
            Udata(i, j, k, dofMap(idx, idy, idz, IV)) = 0.0;
            Udata(i, j, k, dofMap(idx, idy, idz, IW)) = 0.0;
          }

        } // end for idx
      }   // end for idy
    }     // end for idz

  } // end operator () - 3d

  BlastParams bParams;
  DataArray   Udata;

}; // InitBlastFunctor

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_INIT_BLAST_FUNCTOR_H_
