#ifndef SDM_TEST_FUNCTORS_H_
#define SDM_TEST_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h"

#include "shared/EulerEquations.h"

enum data_type_for_test {
  TEST_DATA_VALUE=0,
  TEST_DATA_GRADX=1,
  TEST_DATA_GRADY=2,
  TEST_DATA_GRADZ=3
};

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N, int data_type_test, int compare>
class InitTestFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  InitTestFunctor(HydroParams         params,
		  SDM_Geometry<dim,N> sdm_geom,
		  DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom), Udata(Udata) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    DataArray           Udata)
  {
    int64_t nbDofs = (dim==2) ? 
      params.isize * params.jsize * N * N :
      params.isize * params.jsize * params.ksize * N * N * N;
    
    InitTestFunctor functor(params, sdm_geom, Udata);
    Kokkos::parallel_for("IniTestFunctor", nbDofs, functor);
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
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
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
    
    // global index
    int ii,jj;
    index2coord(index,ii,jj,isize*N,jsize*N);

    // local cell index
    int i = ii/N;
    int j = jj/N;

    // Dof index
    int idx = ii-i*N;
    int idy = jj-j*N;

    // lower left corner
    real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;

    x += this->sdm_geom.solution_pts_1d(idx) * dx;
    y += this->sdm_geom.solution_pts_1d(idy) * dy;
    
    if (compare == 1) {
      Udata(ii, jj, ID) -= f0(x,y,0.0);
      Udata(ii, jj, IP) -= f1(x,y,0.0);
      Udata(ii, jj, IU) -= f2(x,y,0.0);
      Udata(ii, jj, IV) -= f3(x,y,0.0);
    } else {
      Udata(ii, jj, ID) = f0(x,y,0.0);
      Udata(ii, jj, IP) = f1(x,y,0.0);
      Udata(ii, jj, IU) = f2(x,y,0.0);
      Udata(ii, jj, IV) = f3(x,y,0.0);
    }
    
  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
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
    
    // global index
    int ii,jj,kk;
    index2coord(index,ii,jj,kk,isize*N,jsize*N,ksize*N);

    // local cell index
    int i = ii/N;
    int j = jj/N;
    int k = kk/N;

    // Dof index
    int idx = ii-i*N;
    int idy = jj-j*N;
    int idz = kk-k*N;

    // lower left corner
    real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
    real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;
    real_t z = zmin + (k+nz*k_mpi-ghostWidth)*dz;
    
    x += this->sdm_geom.solution_pts_1d(idx) * dx;
    y += this->sdm_geom.solution_pts_1d(idy) * dy;
    z += this->sdm_geom.solution_pts_1d(idz) * dz;
    
    if (compare == 1) {
      Udata(ii, jj, kk, ID) -= f0(x,y,z);
      Udata(ii, jj, kk, IP) -= f1(x,y,z);
      Udata(ii, jj, kk, IU) -= f2(x,y,z);
      Udata(ii, jj, kk, IV) -= f3(x,y,z);
      Udata(ii, jj, kk, IW) -= f4(x,y,z);
    } else {
      Udata(ii, jj, kk, ID) = f0(x,y,z);
      Udata(ii, jj, kk, IP) = f1(x,y,z);
      Udata(ii, jj, kk, IU) = f2(x,y,z);
      Udata(ii, jj, kk, IV) = f3(x,y,z);
      Udata(ii, jj, kk, IW) = f4(x,y,z);
    }
    
  } // end operator () - 3d
  
  DataArray Udata;

}; // InitTestFunctor

} // namespace sdm

#endif // SDM_TEST_FUNCTORS_H_
