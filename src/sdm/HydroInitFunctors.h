#ifndef HYDRO_INIT_FUNCTORS_H_
#define HYDRO_INIT_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

// init conditions
#include "shared/problems/BlastParams.h"
#include "shared/problems/IsentropicVortexParams.h"
#include "shared/problems/initRiemannConfig2d.h"
#include "shared/problems/WedgeParams.h"
#include "shared/problems/JetParams.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitImplodeFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;
  
  InitImplodeFunctor(HydroParams         params,
		     SDM_Geometry<dim,N> sdm_geom,
		     DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom), Udata(Udata) {};

  
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
    
    const real_t gamma0 = this->params.settings.gamma0;

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
	
	real_t tmp = x+y*y;
	if (tmp > 0.5 && tmp < 1.5) {
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) = 1.0;
	  Udata(i  ,j  , dofMap(idx,idy,0,IE)) = 1.0/(gamma0-1.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) = 0.0;
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) = 0.0;
	} else {
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) = 0.125;
	  Udata(i  ,j  , dofMap(idx,idy,0,IE)) = 0.14/(gamma0-1.0);
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) = 0.0;
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) = 0.0;
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
    
    const real_t gamma0 = this->params.settings.gamma0;

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
	  
	  real_t tmp = x+y+z;
	  if (tmp > 0.5 && tmp < 2.5) {
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = 1.0;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IE)) = 1.0/(gamma0-1.0);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = 0.0;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = 0.0;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = 0.0;
	  } else {
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = 0.125;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IE)) = 0.14/(gamma0-1.0);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = 0.0;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = 0.0;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = 0.0;
	  }
	  
	} // end for idx
      } // end for idy
    } // end for idz
    
  } // end operator () - 3d
  
  DataArray Udata;

}; // InitImplodeFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitBlastFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;

  InitBlastFunctor(HydroParams         params,
		   SDM_Geometry<dim,N> sdm_geom,
		   BlastParams         bParams,
		   DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    bParams(bParams),
    Udata(Udata) {};
  
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
    
    const real_t gamma0 = this->params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius      = bParams.blast_radius;
    const real_t radius2           = blast_radius*blast_radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_density_in  = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out= bParams.blast_pressure_out;
    
    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {
	
	// lower left corner
	real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;

	// DoF location
	x += this->sdm_geom.solution_pts_1d(idx) * dx;
	y += this->sdm_geom.solution_pts_1d(idy) * dy;

	real_t d2 = 
	  (x-blast_center_x)*(x-blast_center_x)+
	  (y-blast_center_y)*(y-blast_center_y);    
    
	if (d2 < radius2) {
	  Udata(i,j,dofMap(idx,idy,0,ID)) = blast_density_in;
	  Udata(i,j,dofMap(idx,idy,0,IE)) = blast_pressure_in/(gamma0-1.0);
	  Udata(i,j,dofMap(idx,idy,0,IU)) = 0.0;
	  Udata(i,j,dofMap(idx,idy,0,IV)) = 0.0;
	} else {
	  Udata(i,j,dofMap(idx,idy,0,ID)) = blast_density_out;
	  Udata(i,j,dofMap(idx,idy,0,IE)) = blast_pressure_out/(gamma0-1.0);
	  Udata(i,j,dofMap(idx,idy,0,IU)) = 0.0;
	  Udata(i,j,dofMap(idx,idy,0,IV)) = 0.0;
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
    
    const real_t gamma0 = this->params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius      = bParams.blast_radius;
    const real_t radius2           = blast_radius*blast_radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_center_z    = bParams.blast_center_z;
    const real_t blast_density_in  = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out= bParams.blast_pressure_out;

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

	  real_t d2 = 
	    (x-blast_center_x)*(x-blast_center_x)+
	    (y-blast_center_y)*(y-blast_center_y)+
	    (z-blast_center_z)*(z-blast_center_z);    
    
	  if (d2 < radius2) {

	    Udata(i,j,k,dofMap(idx,idy,idz,ID)) = blast_density_in;
	    Udata(i,j,k,dofMap(idx,idy,idz,IE)) = blast_pressure_in/(gamma0-1.0);
	    Udata(i,j,k,dofMap(idx,idy,idz,IU)) = 0.0;
	    Udata(i,j,k,dofMap(idx,idy,idz,IV)) = 0.0;
	    Udata(i,j,k,dofMap(idx,idy,idz,IW)) = 0.0;
	    
	  } else {

	    Udata(i,j,k,dofMap(idx,idy,idz,ID)) = blast_density_out;
	    Udata(i,j,k,dofMap(idx,idy,idz,IE)) = blast_pressure_out/(gamma0-1.0);
	    Udata(i,j,k,dofMap(idx,idy,idz,IU)) = 0.0;
	    Udata(i,j,k,dofMap(idx,idy,idz,IV)) = 0.0;
	    Udata(i,j,k,dofMap(idx,idy,idz,IW)) = 0.0;

	  }
		
	} // end for idx
      } // end for idy
    } // end for idz
    
  } // end operator () - 3d
  
  BlastParams bParams;
  DataArray   Udata;
  
}; // InitBlastFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitFourQuadrantFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;

  InitFourQuadrantFunctor(HydroParams params,
			  SDM_Geometry<dim,N> sdm_geom,
			  DataArray Udata,
			  HydroState2d U0,
			  HydroState2d U1,
			  HydroState2d U2,
			  HydroState2d U3,
			  real_t xt,
			  real_t yt) :
    SDMBaseFunctor<dim,N>(params,sdm_geom), Udata(Udata),
    U0(U0), U1(U1), U2(U2), U3(U3), xt(xt), yt(yt)
  {};
  
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
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {

	// lower left corner
	real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
	real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

	// Dof location in real space
    	x += this->sdm_geom.solution_pts_1d(idx) * dx;
	y += this->sdm_geom.solution_pts_1d(idy) * dy;

	if (x<xt) {
	  if (y<yt) {
	    // quarter 2
	    Udata(i  ,j  , dofMap(idx,idy,0,ID)) = U2[ID];
	    Udata(i  ,j  , dofMap(idx,idy,0,IE)) = U2[IE];
	    Udata(i  ,j  , dofMap(idx,idy,0,IU)) = U2[IU];
	    Udata(i  ,j  , dofMap(idx,idy,0,IV)) = U2[IV];
	  } else {
	    // quarter 1
	    Udata(i  ,j  , dofMap(idx,idy,0,ID)) = U1[ID];
	    Udata(i  ,j  , dofMap(idx,idy,0,IE)) = U1[IE];
	    Udata(i  ,j  , dofMap(idx,idy,0,IU)) = U1[IU];
	    Udata(i  ,j  , dofMap(idx,idy,0,IV)) = U1[IV];
	  }
	} else {
	  if (y<yt) {
	    // quarter 3
	    Udata(i  ,j  , dofMap(idx,idy,0,ID)) = U3[ID];
	    Udata(i  ,j  , dofMap(idx,idy,0,IE)) = U3[IE];
	    Udata(i  ,j  , dofMap(idx,idy,0,IU)) = U3[IU];
	    Udata(i  ,j  , dofMap(idx,idy,0,IV)) = U3[IV];
	  } else {
	    // quarter 0
	    Udata(i  ,j  , dofMap(idx,idy,0,ID)) = U0[ID];
	    Udata(i  ,j  , dofMap(idx,idy,0,IE)) = U0[IE];
	    Udata(i  ,j  , dofMap(idx,idy,0,IU)) = U0[IU];
	    Udata(i  ,j  , dofMap(idx,idy,0,IV)) = U0[IV];
	  }
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

	} // end for idx
      } // end for idy
    } // end for idz
    
  } // end operator () - 3d

  DataArray Udata;
  HydroState2d U0, U1, U2, U3;
  real_t xt, yt;
  
}; // InitFourQuadrantFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Wedge (double Mach reflection) initial condition functor.
 *
 * See http://amroc.sourceforge.net/examples/euler/2d/html/ramp_n.htm
 *
 */
template<int dim, int N>
class InitWedgeFunctor : public SDMBaseFunctor<dim,N>
{
  
public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  
  static constexpr auto dofMap = DofMap<dim,N>;
  
  InitWedgeFunctor(HydroParams params,
		   SDM_Geometry<dim,N> sdm_geom,
		   WedgeParams wparams,
		   DataArray   Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    wparams(wparams),
    Udata(Udata)
  {};
  
  ~InitWedgeFunctor() {};

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
    
    //const real_t gamma0 = this->params.settings.gamma0;

    const real_t slope_f = this->wparams.slope_f;
    const real_t x_f     = this->wparams.x_f;
    
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
    
	if ( y > slope_f*(x-x_f) ) {
    
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) = wparams.rho1;
	  Udata(i  ,j  , dofMap(idx,idy,0,IE)) = wparams.e_tot1;
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) = wparams.rho_u1;
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) = wparams.rho_v1;
	  
	} else {
	  
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) = wparams.rho2;
	  Udata(i  ,j  , dofMap(idx,idy,0,IE)) = wparams.e_tot2;
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) = wparams.rho_u2;
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) = wparams.rho_v2;
	  
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
    
    //const real_t gamma0 = this->params.settings.gamma0;
    
    const real_t slope_f = this->wparams.slope_f;
    const real_t x_f     = this->wparams.x_f;

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
    
	  if ( y > slope_f*(x-x_f) ) {
	    
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = wparams.rho1;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IE)) = wparams.e_tot1;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = wparams.rho_u1;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = wparams.rho_v1;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = wparams.rho_w1;
	    
	  } else {
	    
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = wparams.rho2;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IE)) = wparams.e_tot2;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = wparams.rho_u2;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = wparams.rho_v2;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = wparams.rho_w2;
	    
	  }

    	} // end for idx
      } // end for idy
    } // end for idz

  } // end operator () - 3d
  
  WedgeParams wparams;
  DataArray   Udata;
  
}; // class InitWedgeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Jet initial condition functor.
 *
 * reference:
 * "On positivity-preserving high order discontinuous Galerkin schemes for
 * compressible Euler equations on rectangular meshes", Xiangxiong Zhang, 
 * Chi-Wang Shu, Journal of Computational Physics, Volume 229, Issue 23,
 * 20 November 2010, Pages 8918-8934
 * http://www.sciencedirect.com/science/article/pii/S0021999110004535
 *
 */
template<int dim, int N>
class InitJetFunctor : public SDMBaseFunctor<dim,N>
{
  
public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  
  static constexpr auto dofMap = DofMap<dim,N>;
  
  InitJetFunctor(HydroParams params,
		 SDM_Geometry<dim,N> sdm_geom,
		 JetParams   jparams,
		 DataArray   Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    jparams(jparams),
    Udata(Udata)
  {};
  
  ~InitJetFunctor() {};

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
        
    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {

	Udata(i  ,j  , dofMap(idx,idy,0,ID)) = jparams.rho2;
	Udata(i  ,j  , dofMap(idx,idy,0,IE)) = jparams.e_tot2;
	Udata(i  ,j  , dofMap(idx,idy,0,IU)) = jparams.rho_u2;
	Udata(i  ,j  , dofMap(idx,idy,0,IV)) = jparams.rho_v2;
	
      } // end for idx
    } // end for idy

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
    
    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    // loop over cell DoF's
    for (int idz=0; idz<N; ++idz) {
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	  
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = jparams.rho2;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IE)) = jparams.e_tot2;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = jparams.rho_u2;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = jparams.rho_v2;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = jparams.rho_w2;

    	} // end for idx
      } // end for idy
    } // end for idz

  } // end operator () - 3d
  
  JetParams   jparams;
  DataArray   Udata;
  
}; // class InitJetFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitIsentropicVortexFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;

  InitIsentropicVortexFunctor(HydroParams            params,
			      SDM_Geometry<dim,N>    sdm_geom,
			      IsentropicVortexParams iparams,
			      DataArray              Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    iparams(iparams),
    Udata(Udata) {};
  
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
    
    const real_t gamma0 = this->params.settings.gamma0;
  
    // ambient flow
    const real_t rho_a = this->iparams.rho_a;
    //const real_t p_a   = this->iparams.p_a;
    const real_t T_a   = this->iparams.T_a;
    const real_t u_a   = this->iparams.u_a;
    const real_t v_a   = this->iparams.v_a;
    //const real_t w_a   = this->iparams.w_a;
    
    // vortex center
    const real_t vortex_x = this->iparams.vortex_x;
    const real_t vortex_y = this->iparams.vortex_y;
    const real_t beta =     this->iparams.beta;

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {
	
	// lower left corner
	real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;

	// DoF location
	x += this->sdm_geom.solution_pts_1d(idx) * dx;
	y += this->sdm_geom.solution_pts_1d(idy) * dy;

	// relative coordinates versus vortex center
	real_t xp = x - vortex_x;
	real_t yp = y - vortex_y;
	real_t r  = sqrt(xp*xp + yp*yp);
	
	real_t du = - yp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
	real_t dv =   xp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
    
	real_t T = T_a - (gamma0-1)*beta*beta/(8*gamma0*M_PI*M_PI)*exp(1.0-r*r);
	real_t rho = rho_a*pow(T/T_a,1.0/(gamma0-1));
	
	Udata(i  ,j  , dofMap(idx,idy,0,ID)) = rho;
	Udata(i  ,j  , dofMap(idx,idy,0,IU)) = rho*(u_a + du);
	Udata(i  ,j  , dofMap(idx,idy,0,IV)) = rho*(v_a + dv);
	//Udata(i  ,j  , dofMap(idx,idy,0,IE)) = pow(rho,gamma0)/(gamma0-1.0) +
	Udata(i  ,j  , dofMap(idx,idy,0,IE)) = rho*T/(gamma0-1.0) +
	  0.5*rho*(u_a + du)*(u_a + du) +
	  0.5*rho*(v_a + dv)*(v_a + dv) ;

      } // end for idx
    } // end for idy

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
    
    const real_t gamma0 = this->params.settings.gamma0;

    // ambient flow
    const real_t rho_a = this->iparams.rho_a;
    //const real_t p_a   = this->iparams.p_a;
    const real_t T_a   = this->iparams.T_a;
    const real_t u_a   = this->iparams.u_a;
    const real_t v_a   = this->iparams.v_a;
    //const real_t w_a   = this->iparams.w_a;
    
    // vortex center
    const real_t vortex_x = this->iparams.vortex_x;
    const real_t vortex_y = this->iparams.vortex_y;
    const real_t beta =     this->iparams.beta;

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

	  // relative coordinates versus vortex center
	  real_t xp = x - vortex_x;
	  real_t yp = y - vortex_y;
	  real_t r  = sqrt(xp*xp + yp*yp);
	  
	  real_t du = - yp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
	  real_t dv =   xp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
	  
	  real_t T = T_a - (gamma0-1)*beta*beta/(8*gamma0*M_PI*M_PI)*exp(1.0-r*r);
	  real_t rho = rho_a*pow(T/T_a,1.0/(gamma0-1));
	  
	  Udata(i,j,k, dofMap(idx,idy,idz,ID)) = rho;
	  Udata(i,j,k, dofMap(idx,idy,idz,IU)) = rho*(u_a + du);
	  Udata(i,j,k, dofMap(idx,idy,idz,IV)) = rho*(v_a + dv);
	  Udata(i,j,k, dofMap(idx,idy,idz,IW)) = 0.0;
	  Udata(i,j,k, dofMap(idx,idy,idz,IE)) = rho*T/(gamma0-1.0) +
	    0.5*rho*(u_a + du)*(u_a + du) +
	    0.5*rho*(v_a + dv)*(v_a + dv) ;
	  
	} // end for idx
      } // end for idy
    } // end for idz
    
  } // end operator () - 3d

  IsentropicVortexParams iparams;
  DataArray Udata;
  
}; // InitIsentropicVortexFunctor
  
} // namespace sdm

#endif // HYDRO_INIT_FUNCTORS_H_
