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

  InitImplodeFunctor(HydroParams params,
		     DataArray   Udata) :
    MoodBaseFunctor<dim,degree>(params), Udata(Udata)  {};
  
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
    
    real_t tmp = x+y*y;
    if (tmp > 0.5 && tmp < 1.5) {
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

  InitBlastFunctor(HydroParams params,
		   BlastParams bParams,
		   DataArray   Udata) :
    MoodBaseFunctor<dim,degree>(params),
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

  InitFourQuadrantFunctor(HydroParams params,
			  DataArray   Udata,
			  HydroState U0,
			  HydroState U1,
			  HydroState U2,
			  HydroState U3,
			  real_t xt,
			  real_t yt) :
    MoodBaseFunctor<dim,degree>(params),
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
    real_t z = zmin + dz/2 + (k-ghostWidth)*dz;

    // TODO - TODO - TODO
    Udata(i  ,j  ,k  , ID) = 1.0;
    Udata(i  ,j  ,k  , IP) = 1.0;
    Udata(i  ,j  ,k  , IU) = 0.0;
    Udata(i  ,j  ,k  , IV) = 0.0;
    Udata(i  ,j  ,k  , IW) = 0.0;
    
  } // end operator () - 3d
  
  DataArray  Udata;
  HydroState U0, U1, U2, U3;
  real_t     xt, yt;
  
}; // class InitFourQuadrantFunctor

} // namespace mood

#endif // MOOD_INIT_FUNCTORS_H_
