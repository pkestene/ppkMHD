#ifndef SDM_INIT_KELVIN_HELMHOLTZ_FUNCTOR_H_
#define SDM_INIT_KELVIN_HELMHOLTZ_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/problems/KHParams.h"

// kokkos random numbers
#include <Kokkos_Random.hpp>

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitKelvinHelmholtzFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;

  InitKelvinHelmholtzFunctor(HydroParams         params,
			     SDM_Geometry<dim,N> sdm_geom,
			     KHParams            khParams,
			     DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    khParams(khParams),
    Udata(Udata),
    rand_pool(khParams.seed)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    KHParams            khParams,
                    DataArray           Udata)
  {
    int nbCells = dim==2 ? 
      params.isize*params.jsize : 
      params.isize*params.jsize*params.ksize;

    InitKelvinHelmholtzFunctor functor(params, sdm_geom, khParams, Udata);
    Kokkos::parallel_for("InitKelvinHelmholtzFunctor",nbCells, functor);
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
    //const real_t xmax = this->params.xmax;
    const real_t ymax = this->params.ymax;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    const real_t gamma0 = this->params.settings.gamma0;

    // Kelvin-Helmholtz problem parameters
    const real_t d_in      = khParams.d_in;
    const real_t d_out     = khParams.d_out;
    const real_t vflow_in  = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl      = khParams.amplitude;
    const real_t pressure  = khParams.pressure;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // get random number state
    rand_type rand_gen = rand_pool.get_state();

    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {
	
	// lower left corner
	real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;

	// DoF location
	x += this->sdm_geom.solution_pts_1d(idx) * dx;
	y += this->sdm_geom.solution_pts_1d(idy) * dy;

	// normalized coordinates in [0,1]
	//real_t xn = (x-xmin)/(xmax-xmin);
	real_t yn = (y-ymin)/(ymax-ymin);
	
	if (khParams.p_rand) {
	  
	  real_t d, u, v;

	  if ( yn < 0.25 or yn > 0.75) {

	    d = d_out;
	    u = vflow_out;
	    v = 0.0;
	    
	  } else {
	    
	    d = d_in;
	    u = vflow_in;
	    v = 0.0;
	    
	  }

	  u += ampl * (rand_gen.drand() - 0.5);
	  v += ampl * (rand_gen.drand() - 0.5);
	  
	  Udata(i,j,dofMap(idx,idy,0,ID)) = d;
	  Udata(i,j,dofMap(idx,idy,0,IU)) = d * u;
	  Udata(i,j,dofMap(idx,idy,0,IV)) = d * v;
	  Udata(i,j,dofMap(idx,idy,0,IE)) =
	    pressure/(gamma0-1.0) + 0.5*d*(u*u + v*v);
      
	} else if (khParams.p_sine_rob) {
	  
	  const int    n     = khParams.mode;
	  const real_t w0    = khParams.w0;
	  const real_t delta = khParams.delta;
	  const double y1 = 0.25;
	  const double y2 = 0.75;
	  const double rho1 = d_in;
	  const double rho2 = d_out;
	  const double v1 = vflow_in;
	  const double v2 = vflow_out;
	  
	  const double ramp = 
	    1.0 / ( 1.0 + exp( 2*(y-y1)/delta ) ) +
	    1.0 / ( 1.0 + exp( 2*(y2-y)/delta ) );
	  
	  const real_t d = rho1 + ramp*(rho2-rho1);
	  const real_t u = v1   + ramp*(v2-v1);
	  const real_t v = w0 * sin(n*M_PI*x);
	  
	  Udata(i,j,dofMap(idx,idy,0,ID)) = d;
	  Udata(i,j,dofMap(idx,idy,0,IU)) = d * u;
	  Udata(i,j,dofMap(idx,idy,0,IV)) = d * v;
	  Udata(i,j,dofMap(idx,idy,0,IE)) =
	    pressure / (gamma0-1.0) + 0.5*d*(u*u + v*v);
	  
	}
	
      } // end for idx
    } // end for idy

    // free random number
    rand_pool.free_state(rand_gen);
    
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

    //const real_t xmax = this->params.xmax;
    //const real_t ymax = this->params.ymax;
    const real_t zmax = this->params.zmax;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
    const real_t gamma0 = this->params.settings.gamma0;

    // Kelvin-Helmholtz problem parameters
    const real_t d_in      = khParams.d_in;
    const real_t d_out     = khParams.d_out;
    const real_t vflow_in  = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl      = khParams.amplitude;
    const real_t pressure  = khParams.pressure;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // get random number state
    rand_type rand_gen = rand_pool.get_state();
    
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

	  // normalized coordinates in [0,1]
	  //real_t xn = (x-xmin)/(xmax-xmin);
	  //real_t yn = (y-ymin)/(ymax-ymin);
	  real_t zn = (z-zmin)/(zmax-zmin);
	  
	  if (khParams.p_rand) {
	    
	    real_t d, u, v, w;
	    
	    if ( zn < 0.25 or zn > 0.75) {
	      
	      d = d_out;
	      u = vflow_out;
	      v = 0.0;
	      w = 0.0;
	      
	    } else {
	      
	      d = d_in;
	      u = vflow_in;
	      v = 0.0;
	      w = 0.0;
	      
	    }
	    
	    u += ampl * (rand_gen.drand() - 0.5);
	    v += ampl * (rand_gen.drand() - 0.5);
	    w += ampl * (rand_gen.drand() - 0.5);
	    
	    Udata(i,j,k,dofMap(idx,idy,idz,ID)) = d;
	    Udata(i,j,k,dofMap(idx,idy,idz,IU)) = d * u;
	    Udata(i,j,k,dofMap(idx,idy,idz,IV)) = d * v;
	    Udata(i,j,k,dofMap(idx,idy,idz,IW)) = d * w;
	    Udata(i,j,k,dofMap(idx,idy,idz,IE)) =
	      pressure/(gamma0-1.0) + 0.5*d*(u*u + v*v + w*w);
	    
	  } else if (khParams.p_sine_rob) {
	    
	    const int    n     = khParams.mode;
	    const real_t w0    = khParams.w0;
	    const real_t delta = khParams.delta;

	    const double z1 = 0.25;
	    const double z2 = 0.75;

	    const double rho1 = d_in;
	    const double rho2 = d_out;

	    const double v1x = vflow_in;
	    const double v2x = vflow_out;
	    
	    const double v1y = vflow_in/2;
	    const double v2y = vflow_out/2;
	    
	    const double ramp = 
	      1.0 / ( 1.0 + exp( 2*(z-z1)/delta ) ) +
	      1.0 / ( 1.0 + exp( 2*(z2-z)/delta ) );
	    
	    const real_t d = rho1 + ramp*(rho2-rho1);
	    const real_t u = v1x   + ramp*(v2x-v1x);
	    const real_t v = v1y   + ramp*(v2y-v1y);
	    const real_t w = w0 * sin(n*M_PI*x) * sin(n*M_PI*y);
	    
	    Udata(i,j,k,dofMap(idx,idy,idz,ID)) = d;
	    Udata(i,j,k,dofMap(idx,idy,idz,IU)) = d * u;
	    Udata(i,j,k,dofMap(idx,idy,idz,IV)) = d * v;
	    Udata(i,j,k,dofMap(idx,idy,idz,IW)) = d * w;
	    Udata(i,j,k,dofMap(idx,idy,idz,IE)) =
	      pressure / (gamma0-1.0) + 0.5*d*(u*u + v*v + w*w);
	    
	  }
	  
	} // end for idx
      } // end for idy
    } // end for idz

    // free random number
    rand_pool.free_state(rand_gen);

  } // end operator () - 3d
  
  KHParams  khParams;
  DataArray Udata;
  
  // random number generator
  Kokkos::Random_XorShift64_Pool<Device> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<Device>::generator_type rand_type;

}; // InitKelvinHelmholtzFunctor

} // namespace sdm

#endif // SDM_INIT_KELVIN_HELMHOLTZ_FUNCTOR_H_
