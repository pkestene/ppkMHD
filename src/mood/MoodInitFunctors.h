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
  Kokkos::Random_XorShift64_Pool<DEVICE> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DEVICE>::generator_type rand_type;
  
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
    iparams(iparams)  {};
  
  ~InitIsentropicVortexFunctor() {};

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

    // ambient flow
    const real_t rho_a = this->iparams.rho_a;
    const real_t p_a   = this->iparams.p_a;
    const real_t T_a   = this->iparams.T_a;
    const real_t u_a   = this->iparams.u_a;
    const real_t v_a   = this->iparams.v_a;
    //const real_t w_a   = this->iparams.w_a;
    
    // vortex center
    const real_t vortex_x = this->iparams.vortex_x;
    const real_t vortex_y = this->iparams.vortex_y;

    // relative coordinates versus vortex center
    real_t xp = x - vortex_x;
    real_t yp = y - vortex_y;
    real_t r  = sqrt(xp*xp + yp*yp);
    
    const real_t beta = this->iparams.beta;

    real_t du = - yp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
    real_t dv =   xp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
    
    real_t T = T_a - (gamma0-1)*beta*beta/(8*gamma0*M_PI*M_PI)*exp(1.0-r*r);
    real_t rho = rho_a*pow(T/T_a,1.0/(gamma0-1));
    
    Udata(i  ,j  , ID) = rho;
    Udata(i  ,j  , IU) = rho*(u_a + du);
    Udata(i  ,j  , IV) = rho*(v_a + dv);
    //Udata(i  ,j  , IP) = pow(rho,gamma0)/(gamma0-1.0) +
    Udata(i  ,j  , IP) = rho*T/(gamma0-1.0) +
      0.5*rho*(u_a + du)*(u_a + du) +
      0.5*rho*(v_a + dv)*(v_a + dv) ;
    
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
    
    // ambient flow
    const real_t rho_a = this->iparams.rho_a;
    const real_t p_a   = this->iparams.p_a;
    const real_t T_a   = this->iparams.T_a;
    const real_t u_a   = this->iparams.u_a;
    const real_t v_a   = this->iparams.v_a;
    const real_t w_a   = this->iparams.w_a;
    
    // vortex center
    const real_t vortex_x = this->iparams.vortex_x;
    const real_t vortex_y = this->iparams.vortex_y;

    // relative coordinates versus vortex center
    real_t xp = x - vortex_x;
    real_t yp = y - vortex_y;
    real_t r  = sqrt(xp*xp + yp*yp);
    
    const real_t beta = this->iparams.beta;

    real_t du = - yp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
    real_t dv =   xp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
    
    real_t T = T_a - (gamma0-1)*beta/(8*gamma0*M_PI*M_PI)*exp(1.0-r*r);
    real_t rho = rho_a*pow(T/T_a,1.0/(gamma0-1));
    
    Udata(i  ,j  ,k  , ID) = rho;
    Udata(i  ,j  ,k  , IP) = rho*T/(gamma0-1.0);
    Udata(i  ,j  ,k  , IU) = u_a + du;
    Udata(i  ,j  ,k  , IV) = v_a + dv;
    Udata(i  ,j  ,k  , IW) = w_a;
    
    Udata(i  ,j  ,k  , ID) = rho;
    Udata(i  ,j  ,k  , IU) = rho*(u_a + du);
    Udata(i  ,j  ,k  , IV) = rho*(v_a + dv);
    Udata(i  ,j  ,k  , IV) = rho*(w_a     );
    Udata(i  ,j  ,k  , IP) = pow(rho,gamma0)/(gamma0-1.0) +
      0.5*rho*(u_a + du)*(u_a + du) +
      0.5*rho*(v_a + dv)*(v_a + dv) +
      0.5*rho*(w_a     )*(w_a     );

  } // end operator () - 3d
  
  DataArray              Udata;
  IsentropicVortexParams iparams;
  
}; // class InitIsentropicVortexFunctor

} // namespace mood

#endif // MOOD_INIT_FUNCTORS_H_
