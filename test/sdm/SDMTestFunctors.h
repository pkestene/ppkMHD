#ifndef SDM_TEST_FUNCTORS_H_
#define SDM_TEST_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/EulerEquations.h"

enum data_type_for_test {
  TEST_DATA_VALUE=0,
  TEST_DATA_GRADX=1,
  TEST_DATA_GRADY=2,
  TEST_DATA_GRADZ=3
};

namespace ppkMHD {
namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N, int data_type_test, int compare>
class InitTestFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;

  InitTestFunctor(HydroParams         params,
		  SDM_Geometry<dim,N> sdm_geom,
		  DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom), Udata(Udata) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    DataArray           Udata)
  {
    int64_t nbCells = (dim==2) ?
      params.isize * params.jsize:
      params.isize * params.jsize * params.ksize;

    InitTestFunctor functor(params, sdm_geom, Udata);
    Kokkos::parallel_for("IniTestFunctor", nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  real_t f0(real_t x, real_t y, real_t z) const
  {
    UNUSED(z);
    return x+y+z;
  }

  KOKKOS_INLINE_FUNCTION
  real_t f1(real_t x, real_t y, real_t z) const
  {
    UNUSED(z);
    return x*x;
  }

  KOKKOS_INLINE_FUNCTION
  real_t f2(real_t x, real_t y, real_t z) const
  {
    UNUSED(z);
    return x*x + x*y + y*y + y*z;
  }

  KOKKOS_INLINE_FUNCTION
  real_t f3(real_t x, real_t y, real_t z) const
  {
    UNUSED(z);
    return x + 2 + cos(M_PI*y);
  }

  KOKKOS_INLINE_FUNCTION
  real_t f4(real_t x, real_t y, real_t z) const
  {
    UNUSED(z);
    return x + 2 + sin(M_PI*y);
  }

  /*
   * 2D version.
   */
  //! functor for 2d
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==2, int>::type& index) const
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

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {

	// lower left corner
	real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;

        x += this->sdm_geom.solution_pts_1d(idx) * dx;
	y += this->sdm_geom.solution_pts_1d(idy) * dy;

	if (compare == 1) {
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) -= f0(x,y,0.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IP)) -= f1(x,y,0.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) -= f2(x,y,0.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) -= f3(x,y,0.0);
	} else {
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) = f0(x,y,0.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IP)) = f1(x,y,0.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) = f2(x,y,0.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) = f3(x,y,0.0);
	}

      } // end for idx
    } // end for idy

  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==3, int>::type& index) const
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

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // loop over cell DoF's
    for (int idz=0; idz<N; ++idz) {
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {

	  // lower left corner
	  real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	  real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;
	  real_t z = zmin + (k+nz*k_mpi-ghostWidth)*dz;

	  x += this->sdm_geom.solution_pts_1d(idx) * dx;
	  y += this->sdm_geom.solution_pts_1d(idy) * dy;
	  z += this->sdm_geom.solution_pts_1d(idz) * dz;

	  if (compare == 1) {
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) -= f0(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IP)) -= f1(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) -= f2(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) -= f3(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) -= f4(x,y,z);
	  } else {
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = f0(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IP)) = f1(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = f2(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = f3(x,y,z);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = f4(x,y,z);
	  }

	} // end for idx
      } // end for idy
    } // end for idz

  } // end operator () - 3d

  DataArray Udata;

}; // InitTestFunctor

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_TEST_FUNCTORS_H_
