/**
 * Init conditions functor for mood simulation solver.
 */
#ifndef MOOD_INIT_FUNCTORS_H_
#define MOOD_INIT_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

// mood
#include "mood/mood_shared.h"
#include "mood/MoodBaseFunctor.h"

// init conditions
#include "shared/BlastParams.h"
#include "shared/KHParams.h"
#include "shared/IsentropicVortexParams.h"

// kokkos random numbers
#include <Kokkos_Random.hpp>


namespace mood {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implode functor.
 */
template<int dim, int degree>
class InitImplodeFunctor : public MoodBaseFunctor<dim,degree>
{

public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using MonomMap = typename mood::MonomialMap<dim,degree>::MonomMap;
  
  InitImplodeFunctor(HydroParams params,
		     MonomMap    monomMap,
		     DataArray   Udata) :
    MoodBaseFunctor<dim,degree>(params,monomMap), Udata(Udata)  {};
  
  ~InitImplodeFunctor() {};

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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    const real_t gamma0 = this->params.settings.gamma0;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    
    //real_t tmp = x+y*y;
    real_t tmp = x+y;
    if (tmp > 0.5 /*&& tmp < 1.5*/) {
      Udata(i  ,j  , ID) = 1.0;
      Udata(i  ,j  , IP) = 1.0/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
    } else {
      Udata(i  ,j  , ID) = 0.125;
      Udata(i  ,j  , IP) = 0.14/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
    const real_t gamma0 = this->params.settings.gamma0;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    real_t z = zmin + dz/2 + (k-ghostWidth)*dz;
    
    real_t tmp = x + y + z;
    if (tmp > 0.5 && tmp < 2.5) {
      Udata(i  ,j  ,k  , ID) = 1.0;
      Udata(i  ,j  ,k  , IP) = 1.0/(gamma0-1.0);
      Udata(i  ,j  ,k  , IU) = 0.0;
      Udata(i  ,j  ,k  , IV) = 0.0;
      Udata(i  ,j  ,k  , IW) = 0.0;
    } else {
      Udata(i  ,j  ,k  , ID) = 0.125;
      Udata(i  ,j  ,k  , IP) = 0.14/(gamma0-1.0);
      Udata(i  ,j  ,k  , IU) = 0.0;
      Udata(i  ,j  ,k  , IV) = 0.0;
      Udata(i  ,j  ,k  , IW) = 0.0;	    
    }
    
  } // end operator () - 3d
  
  DataArray        Udata;
  
}; // class InitImplodeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Blast functor.
 */
template<int dim, int degree>
class InitBlastFunctor : public MoodBaseFunctor<dim,degree>
{

public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using MonomMap = typename mood::MonomialMap<dim,degree>::MonomMap;

  InitBlastFunctor(HydroParams params,
		   MonomMap    monomMap,
		   BlastParams bParams,
		   DataArray   Udata) :
    MoodBaseFunctor<dim,degree>(params,monomMap),
    bParams(bParams),
    Udata(Udata)  {};
  
  ~InitBlastFunctor() {};

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

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    
    real_t d2 = 
      (x-blast_center_x)*(x-blast_center_x)+
      (y-blast_center_y)*(y-blast_center_y);    
    
    if (d2 < radius2) {
      Udata(i  ,j  , ID) = blast_density_in;
      Udata(i  ,j  , IP) = blast_pressure_in/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
    } else {
      Udata(i  ,j  , ID) = blast_density_out;
      Udata(i  ,j  , IP) = blast_pressure_out/(gamma0-1.0);
      Udata(i  ,j  , IU) = 0.0;
      Udata(i  ,j  , IV) = 0.0;
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

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    real_t z = zmin + dz/2 + (k-ghostWidth)*dz;
    
    real_t d2 = 
      (x-blast_center_x)*(x-blast_center_x)+
      (y-blast_center_y)*(y-blast_center_y)+
      (z-blast_center_z)*(z-blast_center_z);    
    
    if (d2 < radius2) {
      Udata(i  ,j  ,k  , ID) = blast_density_in;
      Udata(i  ,j  ,k  , IP) = blast_pressure_in/(gamma0-1.0);
      Udata(i  ,j  ,k  , IU) = 0.0;
      Udata(i  ,j  ,k  , IV) = 0.0;
      Udata(i  ,j  ,k  , IW) = 0.0;
    } else {
      Udata(i  ,j  ,k  , ID) = blast_density_out;
      Udata(i  ,j  ,k  , IP) = blast_pressure_out/(gamma0-1.0);
      Udata(i  ,j  ,k  , IU) = 0.0;
      Udata(i  ,j  ,k  , IV) = 0.0;
      Udata(i  ,j  ,k  , IW) = 0.0;
    }
    
  } // end operator () - 3d
  
  DataArray   Udata;
  BlastParams bParams;
  
}; // class InitBlastFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * FourQuadrant functor.
 */
template<int dim, int degree>
class InitFourQuadrantFunctor : public MoodBaseFunctor<dim,degree>
{

public:
  // use Base class typedef types.
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using typename MoodBaseFunctor<dim,degree>::HydroState;
  using MonomMap = typename mood::MonomialMap<dim,degree>::MonomMap;

  InitFourQuadrantFunctor(HydroParams params,
			  MonomMap    monomMap,
			  DataArray   Udata,
			  HydroState2d U0,
			  HydroState2d U1,
			  HydroState2d U2,
			  HydroState2d U3,
			  real_t xt,
			  real_t yt) :
    MoodBaseFunctor<dim,degree>(params,monomMap),
    Udata(Udata),
    U0(U0), U1(U1), U2(U2), U3(U3), xt(xt), yt(yt)
  {};
  
  ~InitFourQuadrantFunctor() {};

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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    
    if (x<xt) {
      if (y<yt) {
	// quarter 2
	Udata(i  ,j  , ID) = U2[ID];
	Udata(i  ,j  , IP) = U2[IP];
	Udata(i  ,j  , IU) = U2[IU];
	Udata(i  ,j  , IV) = U2[IV];
      } else {
	// quarter 1
	Udata(i  ,j  , ID) = U1[ID];
	Udata(i  ,j  , IP) = U1[IP];
	Udata(i  ,j  , IU) = U1[IU];
	Udata(i  ,j  , IV) = U1[IV];
      }
    } else {
      if (y<yt) {
	// quarter 3
	Udata(i  ,j  , ID) = U3[ID];
	Udata(i  ,j  , IP) = U3[IP];
	Udata(i  ,j  , IU) = U3[IU];
	Udata(i  ,j  , IV) = U3[IV];
      } else {
	// quarter 0
	Udata(i  ,j  , ID) = U0[ID];
	Udata(i  ,j  , IP) = U0[IP];
	Udata(i  ,j  , IU) = U0[IU];
	Udata(i  ,j  , IV) = U0[IV];
      }
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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    //real_t z = zmin + dz/2 + (k-ghostWidth)*dz;

    if (x<xt) {
      if (y<yt) {
	// quarter 2
	Udata(i  ,j  ,k  , ID) = U2[ID];
	Udata(i  ,j  ,k  , IP) = U2[IP];
	Udata(i  ,j  ,k  , IU) = U2[IU];
	Udata(i  ,j  ,k  , IV) = U2[IV];
	Udata(i  ,j  ,k  , IW) = 0.0;
      } else {
	// quarter 1
	Udata(i  ,j  ,k  , ID) = U1[ID];
	Udata(i  ,j  ,k  , IP) = U1[IP];
	Udata(i  ,j  ,k  , IU) = U1[IU];
	Udata(i  ,j  ,k  , IV) = U1[IV];
	Udata(i  ,j  ,k  , IW) = 0.0;
      }
    } else {
      if (y<yt) {
	// quarter 3
	Udata(i  ,j  ,k  , ID) = U3[ID];
	Udata(i  ,j  ,k  , IP) = U3[IP];
	Udata(i  ,j  ,k  , IU) = U3[IU];
	Udata(i  ,j  ,k  , IV) = U3[IV];
	Udata(i  ,j  ,k  , IW) = 0.0;
      } else {
	// quarter 0
	Udata(i  ,j  ,k  , ID) = U0[ID];
	Udata(i  ,j  ,k  , IP) = U0[IP];
	Udata(i  ,j  ,k  , IU) = U0[IU];
	Udata(i  ,j  ,k  , IV) = U0[IV];
	Udata(i  ,j  ,k  , IW) = 0.0;
      }
    }
    
  } // end operator () - 3d
  
  DataArray    Udata;
  HydroState2d U0, U1, U2, U3;
  real_t       xt, yt;
  
}; // class InitFourQuadrantFunctor

/**
 * Kelvin-Helmholtz instability initi conditions functor.
 *
 * See http://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
 *
 * See also article by Robertson et al:
 * "Computational Eulerian hydrodynamics and Galilean invariance", 
 * B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).
 *
 */
template<int dim, int degree>
class InitKelvinHelmholtzFunctor : public MoodBaseFunctor<dim,degree>
{

public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using MonomMap = typename mood::MonomialMap<dim,degree>::MonomMap;
  
  InitKelvinHelmholtzFunctor(HydroParams params,
			     MonomMap    monomMap,
			     KHParams    khParams,
			     DataArray   Udata) :
    MoodBaseFunctor<dim,degree>(params,monomMap),
    khParams(khParams),
    Udata(Udata),
    rand_pool(khParams.seed)
  {};
  
  ~InitKelvinHelmholtzFunctor() {};
  
  /*
   * 2D version.
   */
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {

    // get random number state
    rand_type rand_gen = rand_pool.get_state();
    
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;

    const real_t xmax = this->params.xmax;
    const real_t ymax = this->params.ymax;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    const real_t gamma0 = this->params.settings.gamma0;

    const real_t d_in  = khParams.d_in;
    const real_t d_out = khParams.d_out;
    const real_t vflow_in  = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl      = khParams.amplitude;
    const real_t pressure  = khParams.pressure;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;

    // normalized coordinates in [0,1]
    real_t xn = (x-xmin)/(xmax-xmin);
    real_t yn = (y-ymin)/(ymax-ymin);
    
    if (khParams.p_rand) {
      
      if ( yn < 0.25 or yn > 0.75) {

	Udata(i,j,ID) = d_out;
	Udata(i,j,IU) = d_out * (vflow_out + ampl * (rand_gen.drand() - 0.5));
	Udata(i,j,IV) = d_out * (0.0       + ampl * (rand_gen.drand() - 0.5));;
	Udata(i,j,IP) = pressure/(gamma0-1.0) +
	  0.5*(Udata(i,j,IU)*Udata(i,j,IU) +
	       Udata(i,j,IV)*Udata(i,j,IV))/Udata(i,j,ID);
	
      } else {
	
	Udata(i,j,ID) = d_in;
	Udata(i,j,IU) = d_in * (vflow_in + ampl * (rand_gen.drand() - 0.5));
	Udata(i,j,IV) = d_in * (0.0      + ampl * (rand_gen.drand() - 0.5));
	Udata(i,j,IP) = pressure/(gamma0-1.0) +
	  0.5*(Udata(i,j,IU)*Udata(i,j,IU) +
	       Udata(i,j,IV)*Udata(i,j,IV))/Udata(i,j,ID);
	
      }
      
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
      
      Udata(i,j,ID) = rho1 + ramp*(rho2-rho1);
      Udata(i,j,IU) = Udata(i,j,ID) * (v1 + ramp*(v2-v1));
      Udata(i,j,IV) = Udata(i,j,ID) * w0 * sin(n*M_PI*x);
      Udata(i,j,IP) = pressure / (gamma0-1.0) +
	0.5*(Udata(i,j,IU)*Udata(i,j,IU) +
	     Udata(i,j,IV)*Udata(i,j,IV))/Udata(i,j,ID);

    }
   
    // free random number
    rand_pool.free_state(rand_gen);
    
  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    // get random number generator state
    rand_type rand_gen = rand_pool.get_state();

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;
    const real_t xmax = this->params.xmax;
    const real_t ymax = this->params.ymax;
    const real_t zmax = this->params.zmax;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
    const real_t gamma0 = this->params.settings.gamma0;
    
    const real_t d_in  = khParams.d_in;
    const real_t d_out = khParams.d_out;
    const real_t vflow_in  = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl      = khParams.amplitude;
    const real_t pressure  = khParams.pressure;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    real_t z = zmin + dz/2 + (k-ghostWidth)*dz;
    
    // normalized coordinates in [0,1]
    real_t xn = (x-xmin)/(xmax-xmin);
    real_t yn = (y-ymin)/(ymax-ymin);
    real_t zn = (z-zmin)/(zmax-zmin);

    if ( khParams.p_rand) {
      
      if ( zn < 0.25 or zn > 0.75 ) {
	
	Udata(i,j,k,ID) = d_out;
	Udata(i,j,k,IU) = d_out * (vflow_out + ampl * (rand_gen.drand() - 0.5));
	Udata(i,j,k,IV) = d_out * (0.0       + ampl * (rand_gen.drand() - 0.5));;
	Udata(i,j,k,IW) = d_out * (0.0       + ampl * (rand_gen.drand() - 0.5));;
	Udata(i,j,k,IP) = pressure/(gamma0-1.0) +
	  0.5*(Udata(i,j,k,IU)*Udata(i,j,k,IU) +
	       Udata(i,j,k,IV)*Udata(i,j,k,IV) +
	       Udata(i,j,k,IW)*Udata(i,j,k,IW))/Udata(i,j,k,ID);
	
      } else {
	
	Udata(i,j,k,ID) = d_in;
	Udata(i,j,k,IU) = d_in * (vflow_in  + ampl * (rand_gen.drand() - 0.5));
	Udata(i,j,k,IV) = d_in * (0.0       + ampl * (rand_gen.drand() - 0.5));;
	Udata(i,j,k,IW) = d_in * (0.0       + ampl * (rand_gen.drand() - 0.5));;
	Udata(i,j,k,IP) = pressure/(gamma0-1.0) +
	  0.5 * (Udata(i,j,k,IU)*Udata(i,j,k,IU) +
		 Udata(i,j,k,IV)*Udata(i,j,k,IV) +
		 Udata(i,j,k,IW)*Udata(i,j,k,IW) ) / Udata(i,j,k,ID);
	
      }
      
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
      
      Udata(i,j,k,ID) = rho1 + ramp*(rho2-rho1);
      Udata(i,j,k,IU) = Udata(i,j,k,ID) * (v1x + ramp*(v2x-v1x));
      Udata(i,j,k,IV) = Udata(i,j,k,ID) * (v1y + ramp*(v2y-v1y));
      Udata(i,j,k,IW) = Udata(i,j,k,ID) * w0 * sin(n*M_PI*x) * sin(n*M_PI*y);
      Udata(i,j,k,IP) = pressure / (gamma0-1.0) +
	0.5 * (Udata(i,j,k,IU)*Udata(i,j,k,IU) +
	       Udata(i,j,k,IV)*Udata(i,j,k,IV) +
	       Udata(i,j,k,IW)*Udata(i,j,k,IW) ) / Udata(i,j,k,ID);
    }

    // free random number
    rand_pool.free_state(rand_gen);
    
  } // end operator () - 3d
  
  DataArray        Udata;
  KHParams         khParams;

  // random number generator
  Kokkos::Random_XorShift64_Pool<Device> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<Device>::generator_type rand_type;
  
}; // class InitKelvinHelmholtzFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Wedge (double Mach reflection) initial condition functor.
 *
 * See http://amroc.sourceforge.net/examples/euler/2d/html/ramp_n.htm
 *
 */
template<int dim, int degree>
class InitWedgeFunctor : public MoodBaseFunctor<dim,degree>
{
  
public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using MonomMap = typename mood::MonomialMap<dim,degree>::MonomMap;
  
  InitWedgeFunctor(HydroParams params,
		   MonomMap    monomMap,
		   WedgeParams wparams,
		   DataArray   Udata) :
    MoodBaseFunctor<dim,degree>(params,monomMap),
    Udata(Udata),
    wparams(wparams)
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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    const real_t gamma0 = this->params.settings.gamma0;

    const real_t slope_f = this->wparams.slope_f;
    const real_t x_f     = this->wparams.x_f;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    
    if ( y > slope_f*(x-x_f) ) {
    
      Udata(i  ,j  , ID) = wparams.rho1;
      Udata(i  ,j  , IP) = wparams.e_tot1;
      Udata(i  ,j  , IU) = wparams.rho_u1;
      Udata(i  ,j  , IV) = wparams.rho_v1;
      
    } else {
      
      Udata(i  ,j  , ID) = wparams.rho2;
      Udata(i  ,j  , IP) = wparams.e_tot2;
      Udata(i  ,j  , IU) = wparams.rho_u2;
      Udata(i  ,j  , IV) = wparams.rho_v2;
      
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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
    const real_t gamma0 = this->params.settings.gamma0;
    
    const real_t slope_f = this->wparams.slope_f;
    const real_t x_f     = this->wparams.x_f;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    real_t z = zmin + dz/2 + (k-ghostWidth)*dz;
    
    if ( y > slope_f*(x-x_f) ) {
    
      Udata(i  ,j  ,k  , ID) = wparams.rho1;
      Udata(i  ,j  ,k  , IP) = wparams.e_tot1;
      Udata(i  ,j  ,k  , IU) = wparams.rho_u1;
      Udata(i  ,j  ,k  , IV) = wparams.rho_v1;
      Udata(i  ,j  ,k  , IW) = wparams.rho_w1;

    } else {
    
      Udata(i  ,j  ,k  , ID) = wparams.rho2;
      Udata(i  ,j  ,k  , IP) = wparams.e_tot2;
      Udata(i  ,j  ,k  , IU) = wparams.rho_u2;
      Udata(i  ,j  ,k  , IV) = wparams.rho_v2;
      Udata(i  ,j  ,k  , IW) = wparams.rho_w2;

    }
    
  } // end operator () - 3d
  
  DataArray   Udata;
  WedgeParams wparams;
  
}; // class InitWedgeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Isentropic Vortex advection functor.
 *
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 *
 * Gas law: P = rho T = rho^gamma
 *
 * TO BE MODIFIED: use quadrature rule to provide a higher order approximation
 * of the finite volume init condition.
 *
 */
template<int dim, int degree>
class InitIsentropicVortexFunctor : public MoodBaseFunctor<dim,degree>
{

public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using MonomMap = typename mood::MonomialMap<dim,degree>::MonomMap;
  
  InitIsentropicVortexFunctor(HydroParams params,
			      MonomMap    monomMap,
			      IsentropicVortexParams iparams,
			      DataArray   Udata) :
    MoodBaseFunctor<dim,degree>(params,monomMap),
    Udata(Udata),
    iparams(iparams),
    rho_a(iparams.rho_a),
    p_a(iparams.p_a),
    T_a(iparams.T_a),
    u_a(iparams.u_a),
    v_a(iparams.v_a),
    w_a(iparams.w_a),
    vortex_x(iparams.vortex_x),
    vortex_y(iparams.vortex_y),
    beta(iparams.beta),
    gamma0(params.settings.gamma0)
  {};

  ~InitIsentropicVortexFunctor() {};

  KOKKOS_INLINE_FUNCTION
  real_t compute_T(real_t x, real_t y) const
  {
    // relative coordinates versus vortex center
    real_t xp = x - vortex_x;
    real_t yp = y - vortex_y;
    real_t r  = sqrt(xp*xp + yp*yp);

    real_t T = T_a - (gamma0-1)*beta*beta/(8*gamma0*M_PI*M_PI)*exp(1.0-r*r);

    return T;
  } // compute_T

  KOKKOS_INLINE_FUNCTION
  real_t compute_rho(real_t x, real_t y) const
  {
    // compute temperature
    real_t T = compute_T(x,y);
    
    return rho_a*pow(T/T_a,1.0/(gamma0-1));
  } // compute_rho
  
  KOKKOS_INLINE_FUNCTION
  real_t compute_u(real_t x, real_t y) const
  {

    // relative coordinates versus vortex center
    real_t xp = x - vortex_x;
    real_t yp = y - vortex_y;
    real_t r  = sqrt(xp*xp + yp*yp);
    
    real_t du = - yp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));

    return u_a+du;
  } // compute_u

  KOKKOS_INLINE_FUNCTION
  real_t compute_v(real_t x, real_t y) const
  {

    // relative coordinates versus vortex center
    real_t xp = x - vortex_x;
    real_t yp = y - vortex_y;
    real_t r  = sqrt(xp*xp + yp*yp);
    
    real_t dv =   xp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));

    return v_a+dv;
  } // compute_v

  KOKKOS_INLINE_FUNCTION
  void compute_HydroState_with_quadrature(real_t x, real_t y, int nQuadPts,
					  HydroState2d& q) const
  {

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    real_t rho, u, v, e, T;

    // use Gauss-Legendre Quadrature to compute integral over
    // [x-dx/2, x+dx/2]
    // [y-dy/2, y+dy/2]

    if (nQuadPts == 1) {
      // compute rho, rhou, rhov, e

      T   = compute_T(x,y);
      rho = compute_rho(x,y);
      u   = compute_u(x,y);
      v   = compute_v(x,y);
      e   = rho*T/(gamma0-1) + 0.5*rho*(u*u + v*v + w_a*w_a);
      
      q[ID] = rho;
      q[IU] = rho*u;
      q[IV] = rho*v;
      q[IP] = e;

    } else if (nQuadPts == 2) {

      q[ID] = 0.0;
      q[IU] = 0.0;
      q[IV] = 0.0;
      q[IP] = 0.0;
      
      real_t pos_x[2]; //nQuadPts];
      pos_x[0] = x-0.5*dx*1.0/sqrt(3.0);
      pos_x[1] = x+0.5*dx*1.0/sqrt(3.0);
      
      real_t pos_y[2]; //nQuadPts];
      pos_y[0] = y-0.5*dy*1.0/sqrt(3.0);
      pos_y[1] = y+0.5*dy*1.0/sqrt(3.0);

      real_t weights[2] = {0.5, 0.5};

      for (int jj=0; jj<nQuadPts; ++jj)
	for (int ii=0; ii<nQuadPts; ++ii) {

	  T   = compute_T  (pos_x[ii],pos_y[jj]);
	  rho = compute_rho(pos_x[ii],pos_y[jj]);
	  u   = compute_u  (pos_x[ii],pos_y[jj]);
	  v   = compute_v  (pos_x[ii],pos_y[jj]);
	  e   = rho*T/(gamma0-1) + 0.5*rho*(u*u + v*v + w_a*w_a);
	  
	  q[ID] += rho*weights[ii]*weights[jj];
	  q[IU] += rho*u*weights[ii]*weights[jj];
	  q[IV] += rho*v*weights[ii]*weights[jj];
	  q[IP] += e*weights[ii]*weights[jj];
	  
	}
    } else if (nQuadPts == 3) {

      q[ID] = 0.0;
      q[IU] = 0.0;
      q[IV] = 0.0;
      q[IP] = 0.0;
      
      real_t pos_x[3]; //nQuadPts];
      pos_x[0] = x-0.5*dx*sqrt(3.0)/sqrt(5.0);
      pos_x[1] = x;
      pos_x[2] = x+0.5*dx*sqrt(3.0)/sqrt(5.0);
      
      real_t pos_y[3]; //nQuadPts];
      pos_y[0] = y-0.5*dy*sqrt(3.0)/sqrt(5.0);
      pos_y[1] = y;
      pos_y[2] = y+0.5*dy*sqrt(3.0)/sqrt(5.0);

      real_t weights[3] = {2.5/9, 4.0/9, 2.5/9};

      for (int jj=0; jj<nQuadPts; ++jj)
	for (int ii=0; ii<nQuadPts; ++ii) {

	  T   = compute_T  (pos_x[ii],pos_y[jj]);
	  rho = compute_rho(pos_x[ii],pos_y[jj]);
	  u   = compute_u  (pos_x[ii],pos_y[jj]);
	  v   = compute_v  (pos_x[ii],pos_y[jj]);
	  e   = rho*T/(gamma0-1) + 0.5*rho*(u*u + v*v + w_a*w_a);
	  
	  q[ID] += rho  *weights[ii]*weights[jj];
	  q[IU] += rho*u*weights[ii]*weights[jj];
	  q[IV] += rho*v*weights[ii]*weights[jj];
	  q[IP] += e    *weights[ii]*weights[jj];
	  
	}
    } else if (nQuadPts == 4) {

      q[ID] = 0.0;
      q[IU] = 0.0;
      q[IV] = 0.0;
      q[IP] = 0.0;
      
      real_t pos_x[4];
      pos_x[0] = x-0.5*dx*sqrt(3.0/7-2.0/7*sqrt(6.0/5));
      pos_x[1] = x+0.5*dx*sqrt(3.0/7-2.0/7*sqrt(6.0/5));
      pos_x[2] = x-0.5*dx*sqrt(3.0/7+2.0/7*sqrt(6.0/5));
      pos_x[3] = x+0.5*dx*sqrt(3.0/7+2.0/7*sqrt(6.0/5));
      
      real_t pos_y[4];
      pos_y[0] = y-0.5*dy*sqrt(3.0/7-2.0/7*sqrt(6.0/5));
      pos_y[1] = y+0.5*dy*sqrt(3.0/7-2.0/7*sqrt(6.0/5));
      pos_y[2] = y-0.5*dy*sqrt(3.0/7+2.0/7*sqrt(6.0/5));
      pos_y[3] = y+0.5*dy*sqrt(3.0/7+2.0/7*sqrt(6.0/5));
      
      real_t weights[4];
      weights[0] = 0.5*(0.5+sqrt(30.0)/36);
      weights[1] = 0.5*(0.5+sqrt(30.0)/36);
      weights[2] = 0.5*(0.5-sqrt(30.0)/36);
      weights[3] = 0.5*(0.5-sqrt(30.0)/36);
      
      for (int jj=0; jj<nQuadPts; ++jj)
	for (int ii=0; ii<nQuadPts; ++ii) {

	  T   = compute_T  (pos_x[ii],pos_y[jj]);
	  rho = compute_rho(pos_x[ii],pos_y[jj]);
	  u   = compute_u  (pos_x[ii],pos_y[jj]);
	  v   = compute_v  (pos_x[ii],pos_y[jj]);
	  e   = rho*T/(gamma0-1) + 0.5*rho*(u*u + v*v + w_a*w_a);
	  
	  q[ID] += rho  *weights[ii]*weights[jj];
	  q[IU] += rho*u*weights[ii]*weights[jj];
	  q[IV] += rho*v*weights[ii]*weights[jj];
	  q[IP] += e    *weights[ii]*weights[jj];
	  
	}
    }   
    
  } // compute_HydroState_with_quadrature
  
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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
        
    int i,j;
    index2coord(index,i,j,isize,jsize);

    const int nQuadPts = this->iparams.nQuadPts;
    
    // center of current cell
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;

    HydroState2d q;
    compute_HydroState_with_quadrature(x,y,nQuadPts,q);
    
    Udata(i  ,j  , ID) = q[ID];
    Udata(i  ,j  , IU) = q[IU];
    Udata(i  ,j  , IV) = q[IV];
    Udata(i  ,j  , IP) = q[IP];
    
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
    
    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    const int nQuadPts = this->iparams.nQuadPts;

    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    real_t z = zmin + dz/2 + (k-ghostWidth)*dz;
    
    HydroState2d q;
    compute_HydroState_with_quadrature(x,y,nQuadPts,q);
        
    Udata(i  ,j  ,k  , ID) = q[ID];
    Udata(i  ,j  ,k  , IU) = q[IU];
    Udata(i  ,j  ,k  , IV) = q[IV];
    Udata(i  ,j  ,k  , IW) = q[ID]*w_a;
    Udata(i  ,j  ,k  , IP) = q[IW];

  } // end operator () - 3d
  
  DataArray              Udata;
  IsentropicVortexParams iparams;

  // ambient flow
  const real_t rho_a;
  const real_t p_a;
  const real_t T_a;
  const real_t u_a;
  const real_t v_a;
  const real_t w_a;
  
  // vortex center
  const real_t vortex_x;
  const real_t vortex_y;

  const real_t beta;

  // specific heat ratio
  const real_t gamma0;

}; // class InitIsentropicVortexFunctor

} // namespace mood

#endif // MOOD_INIT_FUNCTORS_H_
