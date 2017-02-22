#ifndef HYDRO_RUN_FUNCTORS_3D_H_
#define HYDRO_RUN_FUNCTORS_3D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "HydroBaseFunctor3D.h"

#include "BlastParams.h"

namespace ppkMHD { namespace muscl { namespace hydro3d {

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor3D : public HydroBaseFunctor3D {

public:
  
  ComputeDtFunctor3D(HydroParams params,
		     DataArray Udata) :
    HydroBaseFunctor3D(params),
    Udata(Udata)  {};

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
  } // init

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &index, real_t &invDt) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize - ghostWidth &&
       j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c=0.0;
      real_t vx, vy, vz;
      
      // get local conservative variable
      uLoc.d = Udata(i,j,k,ID);
      uLoc.p = Udata(i,j,k,IP);
      uLoc.u = Udata(i,j,k,IU);
      uLoc.v = Udata(i,j,k,IV);
      uLoc.w = Udata(i,j,k,IW);

      // get primitive variables in current cell
      computePrimitives(&uLoc, &c, &qLoc);
      vx = c+FABS(qLoc.u);
      vy = c+FABS(qLoc.v);
      vz = c+FABS(qLoc.w);

      invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);
      
    }
	    
  } // operator ()


  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  DataArray Udata;
  
}; // ComputeDtFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor3D : public HydroBaseFunctor3D {

public:

  ConvertToPrimitivesFunctor3D(HydroParams params,
			       DataArray Udata,
			       DataArray Qdata) :
    HydroBaseFunctor3D(params), Udata(Udata), Qdata(Qdata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    if(k >= 0 && k < ksize  &&
       j >= 0 && j < jsize  &&
       i >= 0 && i < isize ) {
      
      HydroState uLoc; // conservative variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc.d = Udata(i,j,k,ID);
      uLoc.p = Udata(i,j,k,IP);
      uLoc.u = Udata(i,j,k,IU);
      uLoc.v = Udata(i,j,k,IV);
      uLoc.w = Udata(i,j,k,IW);
      
      // get primitive variables in current cell
      computePrimitives(&uLoc, &c, &qLoc);

      // copy q state in q global
      Qdata(i,j,k,ID) = qLoc.d;
      Qdata(i,j,k,IP) = qLoc.p;
      Qdata(i,j,k,IU) = qLoc.u;
      Qdata(i,j,k,IV) = qLoc.v;
      Qdata(i,j,k,IW) = qLoc.w;
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qdata;
    
}; // ConvertToPrimitivesFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAndStoreFluxesFunctor3D : public HydroBaseFunctor3D {

public:

  ComputeAndStoreFluxesFunctor3D(HydroParams params,
				 DataArray Qdata,
				 DataArray FluxData_x,
				 DataArray FluxData_y,
				 DataArray FluxData_z,
				 real_t dtdx,
				 real_t dtdy,
				 real_t dtdz) :
    HydroBaseFunctor3D(params),
    Qdata(Qdata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y), 
    FluxData_z(FluxData_z), 
    dtdx(dtdx),
    dtdy(dtdy),
    dtdz(dtdz) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    int indexc, index0, index1, index2, index3, index4, index5;

    if(k >= ghostWidth && k <= ksize-ghostWidth  &&
       j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {
      
      // local primitive variables
      HydroState qLoc; // local primitive variables
      
      // local primitive variables in neighbor cell
      HydroState qLocNeighbor;
      
      // local primitive variables in neighborbood
      HydroState qNeighbors_0;
      HydroState qNeighbors_1;
      HydroState qNeighbors_2;
      HydroState qNeighbors_3;
      HydroState qNeighbors_4;
      HydroState qNeighbors_5;
      
      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqZ;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;
      HydroState dqZ_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;
      HydroState flux_z;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      index0 = coord2index(i+1,j  ,k  ,isize,jsize,ksize);
      index1 = coord2index(i-1,j  ,k  ,isize,jsize,ksize);
      index2 = coord2index(i  ,j+1,k  ,isize,jsize,ksize);
      index3 = coord2index(i  ,j-1,k  ,isize,jsize,ksize);
      index4 = coord2index(i  ,j  ,k+1,isize,jsize,ksize);
      index5 = coord2index(i  ,j  ,k-1,isize,jsize,ksize);

      // get primitive variables state vector
      qLoc.d         = Qdata(i  ,j  ,k  , ID);
      qNeighbors_0.d = Qdata(i+1,j  ,k  , ID);
      qNeighbors_1.d = Qdata(i-1,j  ,k  , ID);
      qNeighbors_2.d = Qdata(i  ,j+1,k  , ID);
      qNeighbors_3.d = Qdata(i  ,j-1,k  , ID);
      qNeighbors_4.d = Qdata(i  ,j  ,k+1, ID);
      qNeighbors_5.d = Qdata(i  ,j  ,k-1, ID);
      
      qLoc.p         = Qdata(i  ,j  ,k  , IP);
      qNeighbors_0.p = Qdata(i+1,j  ,k  , IP);
      qNeighbors_1.p = Qdata(i-1,j  ,k  , IP);
      qNeighbors_2.p = Qdata(i  ,j+1,k  , IP);
      qNeighbors_3.p = Qdata(i  ,j-1,k  , IP);
      qNeighbors_4.p = Qdata(i  ,j  ,k+1, IP);
      qNeighbors_5.p = Qdata(i  ,j  ,k-1, IP);
      
      qLoc.u         = Qdata(i  ,j  ,k  , IU);
      qNeighbors_0.u = Qdata(i+1,j  ,k  , IU);
      qNeighbors_1.u = Qdata(i-1,j  ,k  , IU);
      qNeighbors_2.u = Qdata(i  ,j+1,k  , IU);
      qNeighbors_3.u = Qdata(i  ,j-1,k  , IU);
      qNeighbors_4.u = Qdata(i  ,j  ,k+1, IU);
      qNeighbors_5.u = Qdata(i  ,j  ,k-1, IU);
      
      qLoc.v         = Qdata(i  ,j  ,k  , IV);
      qNeighbors_0.v = Qdata(i+1,j  ,k  , IV);
      qNeighbors_1.v = Qdata(i-1,j  ,k  , IV);
      qNeighbors_2.v = Qdata(i  ,j+1,k  , IV);
      qNeighbors_3.v = Qdata(i  ,j-1,k  , IV);
      qNeighbors_4.v = Qdata(i  ,j  ,k+1, IV);
      qNeighbors_5.v = Qdata(i  ,j  ,k-1, IV);
      
      qLoc.w         = Qdata(i  ,j  ,k  , IW);
      qNeighbors_0.w = Qdata(i+1,j  ,k  , IW);
      qNeighbors_1.w = Qdata(i-1,j  ,k  , IW);
      qNeighbors_2.w = Qdata(i  ,j+1,k  , IW);
      qNeighbors_3.w = Qdata(i  ,j-1,k  , IW);
      qNeighbors_4.w = Qdata(i  ,j  ,k+1, IW);
      qNeighbors_5.w = Qdata(i  ,j  ,k-1, IW);
      
      slope_unsplit_hydro_3d(&qLoc, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &qNeighbors_4, &qNeighbors_5,
			     &dqX, &dqY, &dqZ);
	
      // slopes at left neighbor along X
      indexc = coord2index(i-1,j  ,k  ,isize,jsize,ksize);
      index0 = coord2index(i  ,j  ,k  ,isize,jsize,ksize);
      index1 = coord2index(i-2,j  ,k  ,isize,jsize,ksize);
      index2 = coord2index(i-1,j+1,k  ,isize,jsize,ksize);
      index3 = coord2index(i-1,j-1,k  ,isize,jsize,ksize);
      index4 = coord2index(i-1,j  ,k+1,isize,jsize,ksize);
      index5 = coord2index(i-1,j  ,k-1,isize,jsize,ksize);

      qLocNeighbor.d = Qdata(i-1,j  ,k  , ID);
      qNeighbors_0.d = Qdata(i  ,j  ,k  , ID);
      qNeighbors_1.d = Qdata(i-2,j  ,k  , ID);
      qNeighbors_2.d = Qdata(i-1,j+1,k  , ID);
      qNeighbors_3.d = Qdata(i-1,j-1,k  , ID);
      qNeighbors_4.d = Qdata(i-1,j  ,k+1, ID);
      qNeighbors_5.d = Qdata(i-1,j  ,k-1, ID);
      
      qLocNeighbor.p = Qdata(i-1,j  ,k  , IP);
      qNeighbors_0.p = Qdata(i  ,j  ,k  , IP);
      qNeighbors_1.p = Qdata(i-2,j  ,k  , IP);
      qNeighbors_2.p = Qdata(i-1,j+1,k  , IP);
      qNeighbors_3.p = Qdata(i-1,j-1,k  , IP);
      qNeighbors_4.p = Qdata(i-1,j  ,k+1, IP);
      qNeighbors_5.p = Qdata(i-1,j  ,k-1, IP);
      
      qLocNeighbor.u = Qdata(i-1,j  ,k  , IU);
      qNeighbors_0.u = Qdata(i  ,j  ,k  , IU);
      qNeighbors_1.u = Qdata(i-2,j  ,k  , IU);
      qNeighbors_2.u = Qdata(i-1,j+1,k  , IU);
      qNeighbors_3.u = Qdata(i-1,j-1,k  , IU);
      qNeighbors_4.u = Qdata(i-1,j  ,k+1, IU);
      qNeighbors_5.u = Qdata(i-1,j  ,k-1, IU);

      qLocNeighbor.v = Qdata(i-1,j  ,k  , IV);
      qNeighbors_0.v = Qdata(i  ,j  ,k  , IV);
      qNeighbors_1.v = Qdata(i-2,j  ,k  , IV);
      qNeighbors_2.v = Qdata(i-1,j+1,k  , IV);
      qNeighbors_3.v = Qdata(i-1,j-1,k  , IV);
      qNeighbors_4.v = Qdata(i-1,j  ,k+1, IV);
      qNeighbors_5.v = Qdata(i-1,j  ,k-1, IV);

      qLocNeighbor.w = Qdata(i-1,j  ,k  , IW);
      qNeighbors_0.w = Qdata(i  ,j  ,k  , IW);
      qNeighbors_1.w = Qdata(i-2,j  ,k  , IW);
      qNeighbors_2.w = Qdata(i-1,j+1,k  , IW);
      qNeighbors_3.w = Qdata(i-1,j-1,k  , IW);
      qNeighbors_4.w = Qdata(i-1,j  ,k+1, IW);
      qNeighbors_5.w = Qdata(i-1,j  ,k-1, IW);

      slope_unsplit_hydro_3d(&qLocNeighbor, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &qNeighbors_4, &qNeighbors_5,
			     &dqX_neighbor, &dqY_neighbor, &dqZ_neighbor);
      
      //
      // compute reconstructed states at left interface along X
      //
      
      // left interface : right state
      trace_unsplit_3d_along_dir(&qLoc,
				 &dqX, &dqY, &dqZ,
				 dtdx, dtdy, dtdz,
				 FACE_XMIN, &qright);
      
      // left interface : left state
      trace_unsplit_3d_along_dir(&qLocNeighbor,
				 &dqX_neighbor,&dqY_neighbor,&dqZ_neighbor,
				 dtdx, dtdy, dtdz,
				 FACE_XMAX, &qleft);
      
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_x);
	
      //
      // store fluxes X
      //
      FluxData_x(i  ,j  ,k  , ID) = flux_x.d * dtdx;
      FluxData_x(i  ,j  ,k  , IP) = flux_x.p * dtdx;
      FluxData_x(i  ,j  ,k  , IU) = flux_x.u * dtdx;
      FluxData_x(i  ,j  ,k  , IV) = flux_x.v * dtdx;
      FluxData_x(i  ,j  ,k  , IW) = flux_x.w * dtdx;
      
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      indexc = coord2index(i  ,j-1,k  ,isize,jsize,ksize);
      index0 = coord2index(i+1,j-1,k  ,isize,jsize,ksize);
      index1 = coord2index(i-1,j-1,k  ,isize,jsize,ksize);
      index2 = coord2index(i  ,j  ,k  ,isize,jsize,ksize);
      index3 = coord2index(i  ,j-2,k  ,isize,jsize,ksize);
      index4 = coord2index(i  ,j-1,k+1,isize,jsize,ksize);
      index5 = coord2index(i  ,j-1,k-1,isize,jsize,ksize);

      qLocNeighbor.d = Qdata(i  ,j-1,k  , ID);
      qNeighbors_0.d = Qdata(i+1,j-1,k  , ID);
      qNeighbors_1.d = Qdata(i-1,j-1,k  , ID);
      qNeighbors_2.d = Qdata(i  ,j  ,k  , ID);
      qNeighbors_3.d = Qdata(i  ,j-2,k  , ID);
      qNeighbors_4.d = Qdata(i  ,j-1,k+1, ID);
      qNeighbors_5.d = Qdata(i  ,j-1,k-1, ID);
      
      qLocNeighbor.p = Qdata(i  ,j-1,k  , IP);
      qNeighbors_0.p = Qdata(i+1,j-1,k  , IP);
      qNeighbors_1.p = Qdata(i-1,j-1,k  , IP);
      qNeighbors_2.p = Qdata(i  ,j  ,k  , IP);
      qNeighbors_3.p = Qdata(i  ,j-2,k  , IP);
      qNeighbors_4.p = Qdata(i  ,j-1,k+1, IP);
      qNeighbors_5.p = Qdata(i  ,j-1,k-1, IP);
      
      qLocNeighbor.u = Qdata(i  ,j-1,k  , IU);
      qNeighbors_0.u = Qdata(i+1,j-1,k  , IU);
      qNeighbors_1.u = Qdata(i-1,j-1,k  , IU);
      qNeighbors_2.u = Qdata(i  ,j  ,k  , IU);
      qNeighbors_3.u = Qdata(i  ,j-2,k  , IU);
      qNeighbors_4.u = Qdata(i  ,j-1,k+1, IU);
      qNeighbors_5.u = Qdata(i  ,j-1,k-1, IU);

      qLocNeighbor.v = Qdata(i  ,j-1,k  , IV);
      qNeighbors_0.v = Qdata(i+1,j-1,k  , IV);
      qNeighbors_1.v = Qdata(i-1,j-1,k  , IV);
      qNeighbors_2.v = Qdata(i  ,j  ,k  , IV);
      qNeighbors_3.v = Qdata(i  ,j-2,k  , IV);
      qNeighbors_4.v = Qdata(i  ,j-1,k+1, IV);
      qNeighbors_5.v = Qdata(i  ,j-1,k-1, IV);

      qLocNeighbor.w = Qdata(i  ,j-1,k  , IW);
      qNeighbors_0.w = Qdata(i+1,j-1,k  , IW);
      qNeighbors_1.w = Qdata(i-1,j-1,k  , IW);
      qNeighbors_2.w = Qdata(i  ,j  ,k  , IW);
      qNeighbors_3.w = Qdata(i  ,j-2,k  , IW);
      qNeighbors_4.w = Qdata(i  ,j-1,k+1, IW);
      qNeighbors_5.w = Qdata(i  ,j-1,k-1, IW);

      slope_unsplit_hydro_3d(&qLocNeighbor, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &qNeighbors_4, &qNeighbors_5,
			     &dqX_neighbor, &dqY_neighbor, &dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //
	
      // left interface : right state
      trace_unsplit_3d_along_dir(&qLoc,
				 &dqX, &dqY, &dqZ,
				 dtdx, dtdy, dtdz,
				 FACE_YMIN, &qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(&qLocNeighbor,
				 &dqX_neighbor,&dqY_neighbor,&dqZ_neighbor,
				 dtdx, dtdy, dtdz,
				 FACE_YMAX, &qleft);

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft.u) ,&(qleft.v) );
      swapValues(&(qright.u),&(qright.v));
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_y);

      //
      // store fluxes Y
      //
      FluxData_y(i  ,j  ,k  , ID) = flux_y.d * dtdy;
      FluxData_y(i  ,j  ,k  , IP) = flux_y.p * dtdy;
      FluxData_y(i  ,j  ,k  , IU) = flux_y.u * dtdy;
      FluxData_y(i  ,j  ,k  , IV) = flux_y.v * dtdy;
      FluxData_y(i  ,j  ,k  , IW) = flux_y.w * dtdy;
          
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Z !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Z
      indexc = coord2index(i  ,j  ,k-1,isize,jsize,ksize);
      index0 = coord2index(i+1,j  ,k-1,isize,jsize,ksize);
      index1 = coord2index(i-1,j  ,k-1,isize,jsize,ksize);
      index2 = coord2index(i  ,j+1,k-1,isize,jsize,ksize);
      index3 = coord2index(i  ,j-1,k-1,isize,jsize,ksize);
      index4 = coord2index(i  ,j  ,k  ,isize,jsize,ksize);
      index5 = coord2index(i  ,j  ,k-2,isize,jsize,ksize);

      qLocNeighbor.d = Qdata(i  ,j  ,k-1, ID);
      qNeighbors_0.d = Qdata(i+1,j  ,k-1, ID);
      qNeighbors_1.d = Qdata(i-1,j  ,k-1, ID);
      qNeighbors_2.d = Qdata(i  ,j+1,k-1, ID);
      qNeighbors_3.d = Qdata(i  ,j-1,k-1, ID);
      qNeighbors_4.d = Qdata(i  ,j  ,k  , ID);
      qNeighbors_5.d = Qdata(i  ,j  ,k-2, ID);
      
      qLocNeighbor.p = Qdata(i  ,j  ,k-1, IP);
      qNeighbors_0.p = Qdata(i+1,j  ,k-1, IP);
      qNeighbors_1.p = Qdata(i-1,j  ,k-1, IP);
      qNeighbors_2.p = Qdata(i  ,j+1,k-1, IP);
      qNeighbors_3.p = Qdata(i  ,j-1,k-1, IP);
      qNeighbors_4.p = Qdata(i  ,j  ,k  , IP);
      qNeighbors_5.p = Qdata(i  ,j  ,k-2, IP);
      
      qLocNeighbor.u = Qdata(i  ,j  ,k-1, IU);
      qNeighbors_0.u = Qdata(i+1,j  ,k-1, IU);
      qNeighbors_1.u = Qdata(i-1,j  ,k-1, IU);
      qNeighbors_2.u = Qdata(i  ,j+1,k-1, IU);
      qNeighbors_3.u = Qdata(i  ,j-1,k-1, IU);
      qNeighbors_4.u = Qdata(i  ,j  ,k  , IU);
      qNeighbors_5.u = Qdata(i  ,j  ,k-2, IU);

      qLocNeighbor.v = Qdata(i  ,j  ,k-1, IV);
      qNeighbors_0.v = Qdata(i+1,j  ,k-1, IV);
      qNeighbors_1.v = Qdata(i-1,j  ,k-1, IV);
      qNeighbors_2.v = Qdata(i  ,j+1,k-1, IV);
      qNeighbors_3.v = Qdata(i  ,j-1,k-1, IV);
      qNeighbors_4.v = Qdata(i  ,j  ,k  , IV);
      qNeighbors_5.v = Qdata(i  ,j  ,k-2, IV);

      qLocNeighbor.w = Qdata(i  ,j  ,k-1, IW);
      qNeighbors_0.w = Qdata(i+1,j  ,k-1, IW);
      qNeighbors_1.w = Qdata(i-1,j  ,k-1, IW);
      qNeighbors_2.w = Qdata(i  ,j+1,k-1, IW);
      qNeighbors_3.w = Qdata(i  ,j-1,k-1, IW);
      qNeighbors_4.w = Qdata(i  ,j  ,k  , IW);
      qNeighbors_5.w = Qdata(i  ,j  ,k-2, IW);
      
      slope_unsplit_hydro_3d(&qLocNeighbor, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &qNeighbors_4, &qNeighbors_5,
			     &dqX_neighbor, &dqY_neighbor, &dqZ_neighbor);

      //
      // compute reconstructed states at left interface along Z
      //
	
      // left interface : right state
      trace_unsplit_3d_along_dir(&qLoc,
				 &dqX, &dqY, &dqZ,
				 dtdx, dtdy, dtdz,
				 FACE_ZMIN, &qright);

      // left interface : left state
      trace_unsplit_3d_along_dir(&qLocNeighbor,
				 &dqX_neighbor,&dqY_neighbor,&dqZ_neighbor,
				 dtdx, dtdy, dtdz,
				 FACE_ZMAX, &qleft);

      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      swapValues(&(qleft.u) ,&(qleft.w) );
      swapValues(&(qright.u),&(qright.w));
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_z);

      //
      // store fluxes Z
      //
      FluxData_z(i  ,j  ,k  , ID) = flux_z.d * dtdz;
      FluxData_z(i  ,j  ,k  , IP) = flux_z.p * dtdz;
      FluxData_z(i  ,j  ,k  , IU) = flux_z.u * dtdz;
      FluxData_z(i  ,j  ,k  , IV) = flux_z.v * dtdz;
      FluxData_z(i  ,j  ,k  , IW) = flux_z.w * dtdz;
          
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray FluxData_x;
  DataArray FluxData_y;
  DataArray FluxData_z;
  real_t dtdx, dtdy, dtdz;
  
}; // ComputeAndStoreFluxesFunctor3D
  
/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor3D : public HydroBaseFunctor3D {

public:

  UpdateFunctor3D(HydroParams params,
		  DataArray Udata,
		  DataArray FluxData_x,
		  DataArray FluxData_y,
		  DataArray FluxData_z) :
    HydroBaseFunctor3D(params),
    Udata(Udata), 
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    int index2;

    if(k >= ghostWidth && k < ksize-ghostWidth  &&
       j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      Udata(i  ,j  ,k    , ID) +=  FluxData_x(i  ,j  ,k  , ID);
      Udata(i  ,j  ,k    , IP) +=  FluxData_x(i  ,j  ,k  , IP);
      Udata(i  ,j  ,k    , IU) +=  FluxData_x(i  ,j  ,k  , IU);
      Udata(i  ,j  ,k    , IV) +=  FluxData_x(i  ,j  ,k  , IV);
      Udata(i  ,j  ,k    , IW) +=  FluxData_x(i  ,j  ,k  , IW);

      index2 = coord2index(i+1,j  ,k  ,isize,jsize,ksize);
      Udata(i  ,j  ,k    , ID) -=  FluxData_x(i+1,j  ,k   , ID);
      Udata(i  ,j  ,k    , IP) -=  FluxData_x(i+1,j  ,k   , IP);
      Udata(i  ,j  ,k    , IU) -=  FluxData_x(i+1,j  ,k   , IU);
      Udata(i  ,j  ,k    , IV) -=  FluxData_x(i+1,j  ,k   , IV);
      Udata(i  ,j  ,k    , IW) -=  FluxData_x(i+1,j  ,k   , IW);
      
      Udata(i  ,j  ,k    , ID) +=  FluxData_y(i  ,j  ,k    , ID);
      Udata(i  ,j  ,k    , IP) +=  FluxData_y(i  ,j  ,k    , IP);
      Udata(i  ,j  ,k    , IU) +=  FluxData_y(i  ,j  ,k    , IV); //
      Udata(i  ,j  ,k    , IV) +=  FluxData_y(i  ,j  ,k    , IU); //
      Udata(i  ,j  ,k    , IW) +=  FluxData_y(i  ,j  ,k    , IW);
      
      index2 = coord2index(i  ,j+1,k  ,isize,jsize,ksize);
      Udata(i  ,j  ,k    , ID) -=  FluxData_y(i  ,j+1,k  , ID);
      Udata(i  ,j  ,k    , IP) -=  FluxData_y(i  ,j+1,k  , IP);
      Udata(i  ,j  ,k    , IU) -=  FluxData_y(i  ,j+1,k  , IV); //
      Udata(i  ,j  ,k    , IV) -=  FluxData_y(i  ,j+1,k  , IU); //
      Udata(i  ,j  ,k    , IW) -=  FluxData_y(i  ,j+1,k  , IW);

      Udata(i  ,j  ,k    , ID) +=  FluxData_z(i  ,j  ,k    , ID);
      Udata(i  ,j  ,k    , IP) +=  FluxData_z(i  ,j  ,k    , IP);
      Udata(i  ,j  ,k    , IU) +=  FluxData_z(i  ,j  ,k    , IW); //
      Udata(i  ,j  ,k    , IV) +=  FluxData_z(i  ,j  ,k    , IV);
      Udata(i  ,j  ,k    , IW) +=  FluxData_z(i  ,j  ,k    , IU); //
      
      index2 = coord2index(i  ,j  ,k+1,isize,jsize,ksize);
      Udata(i  ,j  ,k    , ID) -=  FluxData_z(i  ,j  ,k+1, ID);
      Udata(i  ,j  ,k    , IP) -=  FluxData_z(i  ,j  ,k+1, IP);
      Udata(i  ,j  ,k    , IU) -=  FluxData_z(i  ,j  ,k+1, IW); //
      Udata(i  ,j  ,k    , IV) -=  FluxData_z(i  ,j  ,k+1, IV);
      Udata(i  ,j  ,k    , IW) -=  FluxData_z(i  ,j  ,k+1, IU); //

    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData_x;
  DataArray FluxData_y;
  DataArray FluxData_z;
  
}; // UpdateFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class UpdateDirFunctor3D : public HydroBaseFunctor3D {

public:

  UpdateDirFunctor3D(HydroParams params,
		     DataArray Udata,
		     DataArray FluxData) :
    HydroBaseFunctor3D(params),
    Udata(Udata), 
    FluxData(FluxData) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j,k, index2;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize-ghostWidth  &&
       j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      if (dir == XDIR) {

	Udata(i  ,j  ,k  , ID) +=  FluxData(i  ,j  ,k  , ID);
	Udata(i  ,j  ,k  , IP) +=  FluxData(i  ,j  ,k  , IP);
	Udata(i  ,j  ,k  , IU) +=  FluxData(i  ,j  ,k  , IU);
	Udata(i  ,j  ,k  , IV) +=  FluxData(i  ,j  ,k  , IV);
	Udata(i  ,j  ,k  , IW) +=  FluxData(i  ,j  ,k  , IW);
	
	index2 = coord2index(i+1,j,k,isize,jsize,ksize);
	Udata(i  ,j  ,k  , ID) -=  FluxData(i+1,j  ,k   , ID);
	Udata(i  ,j  ,k  , IP) -=  FluxData(i+1,j  ,k   , IP);
	Udata(i  ,j  ,k  , IU) -=  FluxData(i+1,j  ,k   , IU);
	Udata(i  ,j  ,k  , IV) -=  FluxData(i+1,j  ,k   , IV);
	Udata(i  ,j  ,k  , IW) -=  FluxData(i+1,j  ,k   , IW);

      } else if (dir == YDIR) {

	Udata(i  ,j  ,k  , ID) +=  FluxData(i  ,j  ,k  , ID);
	Udata(i  ,j  ,k  , IP) +=  FluxData(i  ,j  ,k  , IP);
	Udata(i  ,j  ,k  , IU) +=  FluxData(i  ,j  ,k  , IU);
	Udata(i  ,j  ,k  , IV) +=  FluxData(i  ,j  ,k  , IV);
	Udata(i  ,j  ,k  , IW) +=  FluxData(i  ,j  ,k  , IW);
	
	index2 = coord2index(i,j+1,k,isize,jsize,ksize);
	Udata(i  ,j  ,k  , ID) -=  FluxData(i  ,j+1,k   , ID);
	Udata(i  ,j  ,k  , IP) -=  FluxData(i  ,j+1,k   , IP);
	Udata(i  ,j  ,k  , IU) -=  FluxData(i  ,j+1,k   , IU);
	Udata(i  ,j  ,k  , IV) -=  FluxData(i  ,j+1,k   , IV);
	Udata(i  ,j  ,k  , IW) -=  FluxData(i  ,j+1,k   , IW);

      } else if (dir == ZDIR) {

	Udata(i  ,j  ,k  , ID) +=  FluxData(i  ,j  ,k  , ID);
	Udata(i  ,j  ,k  , IP) +=  FluxData(i  ,j  ,k  , IP);
	Udata(i  ,j  ,k  , IU) +=  FluxData(i  ,j  ,k  , IU);
	Udata(i  ,j  ,k  , IV) +=  FluxData(i  ,j  ,k  , IV);
	Udata(i  ,j  ,k  , IW) +=  FluxData(i  ,j  ,k  , IW);
	
	index2 = coord2index(i,j,k+1,isize,jsize,ksize);
	Udata(i  ,j  ,k  , ID) -=  FluxData(i  ,j  ,k+1 , ID);
	Udata(i  ,j  ,k  , IP) -=  FluxData(i  ,j  ,k+1 , IP);
	Udata(i  ,j  ,k  , IU) -=  FluxData(i  ,j  ,k+1 , IU);
	Udata(i  ,j  ,k  , IV) -=  FluxData(i  ,j  ,k+1 , IV);
	Udata(i  ,j  ,k  , IW) -=  FluxData(i  ,j  ,k+1,  IW);

      }
      
    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData;
  
}; // UpdateDirFunctor3D

    
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor3D : public HydroBaseFunctor3D {
  
public:
  
  ComputeSlopesFunctor3D(HydroParams params,
			 DataArray Qdata,
			 DataArray Slopes_x,
			 DataArray Slopes_y,
			 DataArray Slopes_z) :
    HydroBaseFunctor3D(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y), Slopes_z(Slopes_z) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    int index0, index1, index2, index3, index4, index5;

    if(k >= ghostWidth-1 && k <= ksize-ghostWidth  &&
       j >= ghostWidth-1 && j <= jsize-ghostWidth  &&
       i >= ghostWidth-1 && i <= isize-ghostWidth ) {

      	// local primitive variables
	HydroState qLoc; // local primitive variables

	// local primitive variables in neighborbood
	HydroState qNeighbors_0;
	HydroState qNeighbors_1;
	HydroState qNeighbors_2;
	HydroState qNeighbors_3;
	HydroState qNeighbors_4;
	HydroState qNeighbors_5;

	// Local slopes and neighbor slopes
	HydroState dqX;
	HydroState dqY;
	HydroState dqZ;

	index0 = coord2index(i+1,j  ,k  ,isize,jsize,ksize);
	index1 = coord2index(i-1,j  ,k  ,isize,jsize,ksize);
	index2 = coord2index(i  ,j+1,k  ,isize,jsize,ksize);
	index3 = coord2index(i  ,j-1,k  ,isize,jsize,ksize);
	index4 = coord2index(i  ,j  ,k+1,isize,jsize,ksize);
	index5 = coord2index(i  ,j  ,k-1,isize,jsize,ksize);

	// get primitive variables state vector
	qLoc.d         = Qdata(i  ,j  ,k   , ID);
	qNeighbors_0.d = Qdata(i+1,j  ,k   , ID);
	qNeighbors_1.d = Qdata(i-1,j  ,k   , ID);
	qNeighbors_2.d = Qdata(i  ,j+1,k   , ID);
	qNeighbors_3.d = Qdata(i  ,j-1,k   , ID);
	qNeighbors_4.d = Qdata(i  ,j  ,k+1 , ID);
	qNeighbors_5.d = Qdata(i  ,j  ,k-1 , ID);
	
	qLoc.p         = Qdata(i  ,j  ,k   , IP);
	qNeighbors_0.p = Qdata(i+1,j  ,k   , IP);
	qNeighbors_1.p = Qdata(i-1,j  ,k   , IP);
	qNeighbors_2.p = Qdata(i  ,j+1,k   , IP);
	qNeighbors_3.p = Qdata(i  ,j-1,k   , IP);
	qNeighbors_4.p = Qdata(i  ,j  ,k+1 , IP);
	qNeighbors_5.p = Qdata(i  ,j  ,k-1 , IP);
	
	qLoc.u         = Qdata(i  ,j  ,k   , IU);
	qNeighbors_0.u = Qdata(i+1,j  ,k   , IU);
	qNeighbors_1.u = Qdata(i-1,j  ,k   , IU);
	qNeighbors_2.u = Qdata(i  ,j+1,k   , IU);
	qNeighbors_3.u = Qdata(i  ,j-1,k   , IU);
	qNeighbors_4.u = Qdata(i  ,j  ,k+1 , IU);
	qNeighbors_5.u = Qdata(i  ,j  ,k-1 , IU);
	
	qLoc.v         = Qdata(i  ,j  ,k   , IV);
	qNeighbors_0.v = Qdata(i+1,j  ,k   , IV);
	qNeighbors_1.v = Qdata(i-1,j  ,k   , IV);
	qNeighbors_2.v = Qdata(i  ,j+1,k   , IV);
	qNeighbors_3.v = Qdata(i  ,j-1,k   , IV);
	qNeighbors_4.v = Qdata(i  ,j  ,k+1 , IV);
	qNeighbors_5.v = Qdata(i  ,j  ,k-1 , IV);
	
	qLoc.w         = Qdata(i  ,j  ,k   , IW);
	qNeighbors_0.w = Qdata(i+1,j  ,k   , IW);
	qNeighbors_1.w = Qdata(i-1,j  ,k   , IW);
	qNeighbors_2.w = Qdata(i  ,j+1,k   , IW);
	qNeighbors_3.w = Qdata(i  ,j-1,k   , IW);
	qNeighbors_4.w = Qdata(i  ,j  ,k+1 , IW);
	qNeighbors_5.w = Qdata(i  ,j  ,k-1 , IW);
	
	slope_unsplit_hydro_3d(&qLoc, 
			       &qNeighbors_0, &qNeighbors_1, 
			       &qNeighbors_2, &qNeighbors_3,
			       &qNeighbors_4, &qNeighbors_5,
			       &dqX, &dqY, &dqZ);
	
	// copy back slopes in global arrays
	Slopes_x(i,j,k, ID) = dqX.d;
	Slopes_y(i,j,k, ID) = dqY.d;
	Slopes_z(i,j,k, ID) = dqZ.d;
	
	Slopes_x(i,j,k, IP) = dqX.p;
	Slopes_y(i,j,k, IP) = dqY.p;
	Slopes_z(i,j,k, IP) = dqZ.p;
	
	Slopes_x(i,j,k, IU) = dqX.u;
	Slopes_y(i,j,k, IU) = dqY.u;
	Slopes_z(i,j,k, IU) = dqZ.u;
	
	Slopes_x(i,j,k, IV) = dqX.v;
	Slopes_y(i,j,k, IV) = dqY.v;
	Slopes_z(i,j,k, IV) = dqZ.v;

	Slopes_x(i,j,k, IW) = dqX.w;
	Slopes_y(i,j,k, IW) = dqY.w;
	Slopes_z(i,j,k, IW) = dqZ.w;
      
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray Slopes_x, Slopes_y, Slopes_z;
  
}; // ComputeSlopesFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor3D : public HydroBaseFunctor3D {
  
public:
  
  ComputeTraceAndFluxes_Functor3D(HydroParams params,
				  DataArray Qdata,
				  DataArray Slopes_x,
				  DataArray Slopes_y,
				  DataArray Slopes_z,
				  DataArray Fluxes,
				  real_t    dtdx,
				  real_t    dtdy,
				  real_t    dtdz) :
    HydroBaseFunctor3D(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y), Slopes_z(Slopes_z),
    Fluxes(Fluxes),
    dtdx(dtdx), dtdy(dtdy), dtdz(dtdz) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    int index2;
    
    if(k >= ghostWidth && k <= ksize-ghostWidth  &&
       j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {

	// local primitive variables
	HydroState qLoc; // local primitive variables

	// local primitive variables in neighbor cell
	HydroState qLocNeighbor;

	// Local slopes and neighbor slopes
	HydroState dqX;
	HydroState dqY;
	HydroState dqZ;
	HydroState dqX_neighbor;
	HydroState dqY_neighbor;
	HydroState dqZ_neighbor;

	// Local variables for Riemann problems solving
	HydroState qleft;
	HydroState qright;
	HydroState qgdnv;
	HydroState flux;

	//
	// compute reconstructed states at left interface along X
	//
	qLoc.d = Qdata   (i,j,k, ID);
	dqX.d  = Slopes_x(i,j,k, ID);
	dqY.d  = Slopes_y(i,j,k, ID);
	dqZ.d  = Slopes_z(i,j,k, ID);
	
	qLoc.p = Qdata   (i,j,k, IP);
	dqX.p  = Slopes_x(i,j,k, IP);
	dqY.p  = Slopes_y(i,j,k, IP);
	dqZ.p  = Slopes_z(i,j,k, IP);
	
	qLoc.u = Qdata   (i,j,k, IU);
	dqX.u  = Slopes_x(i,j,k, IU);
	dqY.u  = Slopes_y(i,j,k, IU);
	dqZ.u  = Slopes_z(i,j,k, IU);

	qLoc.v = Qdata   (i,j,k, IV);
	dqX.v  = Slopes_x(i,j,k, IV);
	dqY.v  = Slopes_y(i,j,k, IV);
	dqZ.v  = Slopes_z(i,j,k, IV);

	qLoc.w = Qdata   (i,j,k, IW);
	dqX.w  = Slopes_x(i,j,k, IW);
	dqY.w  = Slopes_y(i,j,k, IW);
	dqZ.w  = Slopes_z(i,j,k, IW);

	if (dir == XDIR) {

	  index2 = coord2index(i-1,j,k,isize,jsize,ksize);

	  // left interface : right state
	  trace_unsplit_3d_along_dir(&qLoc,
				     &dqX, &dqY, &dqZ,
				     dtdx, dtdy, dtdz,
				     FACE_XMIN, &qright);
	  
	  qLocNeighbor.d = Qdata   (i-1,j  ,k  , ID);
	  dqX_neighbor.d = Slopes_x(i-1,j  ,k  , ID);
	  dqY_neighbor.d = Slopes_y(i-1,j  ,k  , ID);
	  dqZ_neighbor.d = Slopes_z(i-1,j  ,k  , ID);
	  
	  qLocNeighbor.p = Qdata   (i-1,j  ,k  , IP);
	  dqX_neighbor.p = Slopes_x(i-1,j  ,k  , IP);
	  dqY_neighbor.p = Slopes_y(i-1,j  ,k  , IP);
	  dqZ_neighbor.p = Slopes_z(i-1,j  ,k  , IP);
	  
	  qLocNeighbor.u = Qdata   (i-1,j  ,k  , IU);
	  dqX_neighbor.u = Slopes_x(i-1,j  ,k  , IU);
	  dqY_neighbor.u = Slopes_y(i-1,j  ,k  , IU);
	  dqZ_neighbor.u = Slopes_z(i-1,j  ,k  , IU);
	  
	  qLocNeighbor.v = Qdata   (i-1,j  ,k  , IV);
	  dqX_neighbor.v = Slopes_x(i-1,j  ,k  , IV);
	  dqY_neighbor.v = Slopes_y(i-1,j  ,k  , IV);
	  dqZ_neighbor.v = Slopes_z(i-1,j  ,k  , IV);
	  
	  qLocNeighbor.w = Qdata   (i-1,j  ,k  , IW);
	  dqX_neighbor.w = Slopes_x(i-1,j  ,k  , IW);
	  dqY_neighbor.w = Slopes_y(i-1,j  ,k  , IW);
	  dqZ_neighbor.w = Slopes_z(i-1,j  ,k  , IW);
	  
	  // left interface : left state
	  trace_unsplit_3d_along_dir(&qLocNeighbor,
				     &dqX_neighbor,&dqY_neighbor,&dqZ_neighbor,
				     dtdx, dtdy, dtdz,
				     FACE_XMAX, &qleft);
	  
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann_hllc(&qleft,&qright,&qgdnv,&flux);

	  //
	  // store fluxes
	  //	
	  Fluxes(i  ,j  ,k  , ID) =  flux.d*dtdx;
	  Fluxes(i  ,j  ,k  , IP) =  flux.p*dtdx;
	  Fluxes(i  ,j  ,k  , IU) =  flux.u*dtdx;
	  Fluxes(i  ,j  ,k  , IV) =  flux.v*dtdx;
	  Fluxes(i  ,j  ,k  , IW) =  flux.w*dtdx;

	} else if (dir == YDIR) {

	  index2 = coord2index(i,j-1,k,isize,jsize,ksize);

	  // left interface : right state
	  trace_unsplit_3d_along_dir(&qLoc,
				     &dqX, &dqY, &dqZ,
				     dtdx, dtdy, dtdz,
				     FACE_YMIN, &qright);
	  
	  qLocNeighbor.d = Qdata   (i  ,j-1,k  , ID);
	  dqX_neighbor.d = Slopes_x(i  ,j-1,k  , ID);
	  dqY_neighbor.d = Slopes_y(i  ,j-1,k  , ID);
	  dqZ_neighbor.d = Slopes_z(i  ,j-1,k  , ID);
	  
	  qLocNeighbor.p = Qdata   (i  ,j-1,k  , IP);
	  dqX_neighbor.p = Slopes_x(i  ,j-1,k  , IP);
	  dqY_neighbor.p = Slopes_y(i  ,j-1,k  , IP);
	  dqZ_neighbor.p = Slopes_z(i  ,j-1,k  , IP);
	  
	  qLocNeighbor.u = Qdata   (i  ,j-1,k  , IU);
	  dqX_neighbor.u = Slopes_x(i  ,j-1,k  , IU);
	  dqY_neighbor.u = Slopes_y(i  ,j-1,k  , IU);
	  dqZ_neighbor.u = Slopes_z(i  ,j-1,k  , IU);

	  qLocNeighbor.v = Qdata   (i  ,j-1,k  , IV);
	  dqX_neighbor.v = Slopes_x(i  ,j-1,k  , IV);
	  dqY_neighbor.v = Slopes_y(i  ,j-1,k  , IV);
	  dqZ_neighbor.v = Slopes_z(i  ,j-1,k  , IV);

	  qLocNeighbor.w = Qdata   (i  ,j-1,k  , IW);
	  dqX_neighbor.w = Slopes_x(i  ,j-1,k  , IW);
	  dqY_neighbor.w = Slopes_y(i  ,j-1,k  , IW);
	  dqZ_neighbor.w = Slopes_z(i  ,j-1,k  , IW);

	  // left interface : left state
	  trace_unsplit_3d_along_dir(&qLocNeighbor,
				     &dqX_neighbor,&dqY_neighbor,&dqZ_neighbor,
				     dtdx, dtdy, dtdz,
				     FACE_YMAX, &qleft);
	  
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  swapValues(&(qleft.u) ,&(qleft.v) );
	  swapValues(&(qright.u),&(qright.v));
	  riemann_hllc(&qleft,&qright,&qgdnv,&flux);
	  
	  //
	  // update hydro array
	  //	  
	  Fluxes(i  ,j  ,k  , ID) =  flux.d*dtdy;
	  Fluxes(i  ,j  ,k  , IP) =  flux.p*dtdy;
	  Fluxes(i  ,j  ,k  , IU) =  flux.v*dtdy; // IU/IV swapped
	  Fluxes(i  ,j  ,k  , IV) =  flux.u*dtdy; // IU/IV swapped
	  Fluxes(i  ,j  ,k  , IW) =  flux.w*dtdy;

	} else if (dir == ZDIR) {

	  index2 = coord2index(i,j,k-1,isize,jsize,ksize);

	  // left interface : right state
	  trace_unsplit_3d_along_dir(&qLoc,
				     &dqX, &dqY, &dqZ,
				     dtdx, dtdy, dtdz,
				     FACE_ZMIN, &qright);
	  
	  qLocNeighbor.d = Qdata   (i  ,j  ,k-1  , ID);
	  dqX_neighbor.d = Slopes_x(i  ,j  ,k-1  , ID);
	  dqY_neighbor.d = Slopes_y(i  ,j  ,k-1  , ID);
	  dqZ_neighbor.d = Slopes_z(i  ,j  ,k-1  , ID);
	  
	  qLocNeighbor.p = Qdata   (i  ,j  ,k-1  , IP);
	  dqX_neighbor.p = Slopes_x(i  ,j  ,k-1  , IP);
	  dqY_neighbor.p = Slopes_y(i  ,j  ,k-1  , IP);
	  dqZ_neighbor.p = Slopes_z(i  ,j  ,k-1  , IP);
	  
	  qLocNeighbor.u = Qdata   (i  ,j  ,k-1  , IU);
	  dqX_neighbor.u = Slopes_x(i  ,j  ,k-1  , IU);
	  dqY_neighbor.u = Slopes_y(i  ,j  ,k-1  , IU);
	  dqZ_neighbor.u = Slopes_z(i  ,j  ,k-1  , IU);

	  qLocNeighbor.v = Qdata   (i  ,j  ,k-1  , IV);
	  dqX_neighbor.v = Slopes_x(i  ,j  ,k-1  , IV);
	  dqY_neighbor.v = Slopes_y(i  ,j  ,k-1  , IV);
	  dqZ_neighbor.v = Slopes_z(i  ,j  ,k-1  , IV);

	  qLocNeighbor.w = Qdata   (i  ,j  ,k-1  , IW);
	  dqX_neighbor.w = Slopes_x(i  ,j  ,k-1  , IW);
	  dqY_neighbor.w = Slopes_y(i  ,j  ,k-1  , IW);
	  dqZ_neighbor.w = Slopes_z(i  ,j  ,k-1  , IW);

	  // left interface : left state
	  trace_unsplit_3d_along_dir(&qLocNeighbor,
				     &dqX_neighbor,&dqY_neighbor,&dqZ_neighbor,
				     dtdx, dtdy, dtdz,
				     FACE_ZMAX, &qleft);
	  
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  swapValues(&(qleft.u) ,&(qleft.w) );
	  swapValues(&(qright.u),&(qright.w));
	  riemann_hllc(&qleft,&qright,&qgdnv,&flux);
	  
	  //
	  // update hydro array
	  //	  
	  Fluxes(i  ,j  ,k  , ID) =  flux.d*dtdz;
	  Fluxes(i  ,j  ,k  , IP) =  flux.p*dtdz;
	  Fluxes(i  ,j  ,k  , IU) =  flux.w*dtdz; // IU/IW swapped
	  Fluxes(i  ,j  ,k  , IV) =  flux.v*dtdz;
	  Fluxes(i  ,j  ,k  , IW) =  flux.u*dtdz; // IU/IW swapped

	}
	      
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray Slopes_x, Slopes_y, Slopes_z;
  DataArray Fluxes;
  real_t dtdx, dtdy, dtdz;
  
}; // ComputeTraceAndFluxes_Functor3D

    
/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor3D : public HydroBaseFunctor3D {

public:
  InitImplodeFunctor3D(HydroParams params,
		       DataArray Udata) :
    HydroBaseFunctor3D(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;
    
    const real_t gamma0 = params.settings.gamma0;
    
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
    
  } // end operator ()

  DataArray Udata;

}; // InitImplodeFunctor3D
  
/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor3D : public HydroBaseFunctor3D {

public:
  InitBlastFunctor3D(HydroParams params,
		     BlastParams bParams,
		     DataArray Udata) :
    HydroBaseFunctor3D(params), bParams(bParams), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;
    
    const real_t gamma0 = params.settings.gamma0;

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
    
  } // end operator ()
  
  DataArray Udata;
  BlastParams bParams;
  
}; // InitBlastFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
template <FaceIdType faceId>
class MakeBoundariesFunctor3D : public HydroBaseFunctor3D {

public:

  MakeBoundariesFunctor3D(HydroParams params,
			DataArray Udata) :
    HydroBaseFunctor3D(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;
    
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    const int imin = params.imin;
    const int imax = params.imax;
    
    const int jmin = params.jmin;
    const int jmax = params.jmax;

    const int kmin = params.kmin;
    const int kmax = params.kmax;
    
    int i,j,k;
    
    int boundary_type;
    
    int i0, j0, k0;
    int iVar;
    int index_out, index_in;
    
    if (faceId == FACE_XMIN) {
      
      // boundary xmin (index = i + j * ghostWidth + k * ghostWidth*jsize)
      k = index / (ghostWidth*jsize);
      j = (index - k*ghostWidth*jsize) / ghostWidth;
      i = index - j*ghostWidth - k*ghostWidth*jsize;
      
      boundary_type = params.boundary_type_xmin;

      if(k >= kmin && k <= kmax &&
	 j >= jmin && j <= jmax &&
	 i >= 0    && i <ghostWidth) {
	
	real_t sign=1.0;
	for ( iVar=0; iVar<nbvar; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*ghostWidth-1-i;
	    if (iVar==IU) sign=-ONE_F;
	  } else if( boundary_type == BC_NEUMANN ) {
	    i0=ghostWidth;
	} else { // periodic
	    i0=nx+i;
	  }
	  
	  index_out = coord2index(i ,j,k,isize,jsize,ksize);
	  index_in  = coord2index(i0,j,k,isize,jsize,ksize);
	  Udata(i,j,k, iVar) = Udata(i0,j,k, iVar)*sign;
	  
	}
	
      } // end xmin
    }

    if (faceId == FACE_XMAX) {
      
      // boundary xmax (index = i + j *ghostWidth + k * ghostWidth*jsize)
      // same i,j,k as xmin, except translation along x-axis
      k = index / (ghostWidth*jsize);
      j = (index - k*ghostWidth*jsize) / ghostWidth;
      i = index - j*ghostWidth - k*ghostWidth*jsize;

      i += (nx+ghostWidth);
      
      boundary_type = params.boundary_type_xmax;
      
      if(k >= kmin          && k <= kmax &&
	 j >= jmin          && j <= jmax &&
	 i >= nx+ghostWidth && i <= nx+2*ghostWidth-1) {
	
	real_t sign=1.0;
	for ( iVar=0; iVar<nbvar; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*nx+2*ghostWidth-1-i;
	    if (iVar==IU) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    i0=nx+ghostWidth-1;
	  } else { // periodic
	    i0=i-nx;
	  }
	  
	  index_out = coord2index(i ,j,k,isize,jsize,ksize);
	  index_in  = coord2index(i0,j,k,isize,jsize,ksize);
	  Udata(i,j,k, iVar) = Udata(i0,j,k, iVar)*sign;
	  
	}
      } // end xmax
    }

    if (faceId == FACE_YMIN) {

      // boundary ymin (index = i + j*isize + k*isize*ghostWidth)
      k = index / (isize*ghostWidth);
      j = (index - k*isize*ghostWidth) / isize;
      i = index - j*isize - k*isize*ghostWidth;

      boundary_type = params.boundary_type_ymin;
      
      if(k >= kmin && k <= kmax       && 
	 j >= 0    && j <  ghostWidth &&
	 i >= imin && i <= imax) {
	
	real_t sign=1.0;
	
	for ( iVar=0; iVar<nbvar; iVar++ ) {
	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ghostWidth-1-j;
	    if (iVar==IV) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ghostWidth;
	  } else { // periodic
	    j0=ny+j;
	  }
	  
	  index_out = coord2index(i,j ,k,isize,jsize,ksize);
	  index_in  = coord2index(i,j0,k,isize,jsize,ksize);
	  Udata(i,j,k, iVar) = Udata(i,j0,k, iVar)*sign;
	  
	}
      } // end ymin
    }

    if (faceId == FACE_YMAX) {
      
      // boundary ymax (index = i + j*isize + k*isize*ghostWidth)
      // same i,j,k as ymin, except translation along y-axis
      k = index / (isize*ghostWidth);
      j = (index - k*isize*ghostWidth) / isize;
      i = index - j*isize - k*isize*ghostWidth;

      j += (ny+ghostWidth);

      boundary_type = params.boundary_type_ymax;

      if(k >= kmin           && k <= kmax              &&
	 j >= ny+ghostWidth  && j <= ny+2*ghostWidth-1 &&
	 i >= imin           && i <= imax) {
	
	real_t sign=1.0;
	for ( iVar=0; iVar<nbvar; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ny+2*ghostWidth-1-j;
	    if (iVar==IV) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ny+ghostWidth-1;
	  } else { // periodic
	    j0=j-ny;
	  }
	  
	  index_out = coord2index(i,j ,k,isize,jsize,ksize);
	  index_in  = coord2index(i,j0,k,isize,jsize,ksize);
	  Udata(i,j,k, iVar) = Udata(i,j0,k, iVar)*sign;
	  
	}
	
      } // end ymax
    }

    if (faceId == FACE_ZMIN) {
      
      // boundary zmin (index = i + j*isize + k*isize*jsize)
      k = index / (isize*jsize);
      j = (index - k*isize*jsize) / isize;
      i = index - j*isize - k*isize*jsize;

      boundary_type = params.boundary_type_zmin;
      
      if(k >= 0    && k <  ghostWidth &&
	 j >= jmin && j <= jmax       &&
	 i >= imin && i <= imax) {
	
	real_t sign=1.0;
	
	for ( iVar=0; iVar<nbvar; iVar++ ) {
	  if ( boundary_type == BC_DIRICHLET ) {
	    k0=2*ghostWidth-1-k;
	    if (iVar==IW) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    k0=ghostWidth;
	  } else { // periodic
	    k0=nz+k;
	  }
	  
	  index_out = coord2index(i,j,k ,isize,jsize,ksize);
	  index_in  = coord2index(i,j,k0,isize,jsize,ksize);
	  Udata(i,j,k, iVar) = Udata(i,j,k0, iVar)*sign;
	  
	}
      } // end zmin
    }
    
    if (faceId == FACE_ZMAX) {
      
      // boundary zmax (index = i + j*isize + k*isize*jsize)
      // same i,j,k as ymin, except translation along y-axis
      k = index / (isize*jsize);
      j = (index - k*isize*jsize) / isize;
      i = index - j*isize - k*isize*jsize;

      k += (nz+ghostWidth);

      boundary_type = params.boundary_type_zmax;

      if(k >= nz+ghostWidth && k <= nz+2*ghostWidth-1 &&
	 j >= jmin          && j <= jmax              &&
	 i >= imin          && i <= imax) {
	
	real_t sign=1.0;
	
	for ( iVar=0; iVar<nbvar; iVar++ ) {
	  if ( boundary_type == BC_DIRICHLET ) {
	    k0=2*nz+2*ghostWidth-1-k;
	    if (iVar==IW) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    k0=nz+ghostWidth-1;
	  } else { // periodic
	    k0=k-nz;
	  }
	  
	  index_out = coord2index(i,j,k ,isize,jsize,ksize);
	  index_in  = coord2index(i,j,k0,isize,jsize,ksize);
	  Udata(i,j,k, iVar) = Udata(i,j,k0, iVar)*sign;
	  
	}
      } // end zmax
    }
    
  } // end operator ()

  DataArray Udata;
  
}; // MakeBoundariesFunctor3D
  
} // namespace hydro3d
} // namespace muscl
} // namespace ppkMHD

#endif // HYDRO_RUN_FUNCTORS_3D_H_

