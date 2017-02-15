#ifndef HYDRO_RUN_FUNCTORS_H_
#define HYDRO_RUN_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__


#include "HydroBaseFunctor2D.h"

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor : public HydroBaseFunctor2D {

public:
  
  ComputeDtFunctor(HydroParams params,
		   DataArray Udata) :
    HydroBaseFunctor2D(params),
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
    const int ghostWidth = params.ghostWidth;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c=0.0;
      real_t vx, vy;
      
      // get local conservative variable
      uLoc.d = Udata(i,j,ID);
      uLoc.p = Udata(i,j,IP);
      uLoc.u = Udata(i,j,IU);
      uLoc.v = Udata(i,j,IV);

      // get primitive variables in current cell
      computePrimitives(&uLoc, &c, &qLoc);
      vx = c+FABS(qLoc.u);
      vy = c+FABS(qLoc.v);

      invDt = FMAX(invDt, vx/dx + vy/dy);
      
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
  
}; // ComputeDtFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor : public HydroBaseFunctor2D {

public:

  ConvertToPrimitivesFunctor(HydroParams params,
			     DataArray Udata,
			     DataArray Qdata) :
    HydroBaseFunctor2D(params), Udata(Udata), Qdata(Qdata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    //const int ghostWidth = params.ghostWidth;
        
    if(j >= 0 && j < jsize  &&
       i >= 0 && i < isize ) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc.d = Udata(i,j,ID);
      uLoc.p = Udata(i,j,IP);
      uLoc.u = Udata(i,j,IU);
      uLoc.v = Udata(i,j,IV);
      
      // get primitive variables in current cell
      computePrimitives(&uLoc, &c, &qLoc);

      // copy q state in q global
      Qdata(i,j,ID) = qLoc.d;
      Qdata(i,j,IP) = qLoc.p;
      Qdata(i,j,IU) = qLoc.u;
      Qdata(i,j,IV) = qLoc.v;
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qdata;
    
}; // ConvertToPrimitivesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndUpdateFunctor : public HydroBaseFunctor2D {

public:

  ComputeFluxesAndUpdateFunctor(HydroParams params,
				DataArray Udata,
				DataArray Qm_x,
				DataArray Qm_y,
				DataArray Qp_x,
				DataArray Qp_y,
				real_t dtdx,
				real_t dtdy) :
    HydroBaseFunctor2D(params), Udata(Udata),
    Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    if(j >= ghostWidth && j <= jsize - ghostWidth &&
       i >= ghostWidth && i <= isize - ghostWidth) {
      
      HydroState qleft, qright;
      HydroState flux_x, flux_y;
      HydroState qgdnv;

      //
      // Solve Riemann problem at X-interfaces and compute
      // X-fluxes
      //
      qleft.d   = Qm_x(i-1,j  , ID);
      qleft.p   = Qm_x(i-1,j  , IP);
      qleft.u   = Qm_x(i-1,j  , IU);
      qleft.v   = Qm_x(i-1,j  , IV);
      
      qright.d  = Qp_x(i  ,j  , ID);
      qright.p  = Qp_x(i  ,j  , IP);
      qright.u  = Qp_x(i  ,j  , IU);
      qright.v  = Qp_x(i  ,j  , IV);
      
      // compute hydro flux_x
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_x);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      qleft.d   = Qm_y(i  ,j-1, ID);
      qleft.p   = Qm_y(i  ,j-1, IP);
      qleft.u   = Qm_y(i  ,j-1, IV); // watchout IU, IV permutation
      qleft.v   = Qm_y(i  ,j-1, IU); // watchout IU, IV permutation

      qright.d  = Qp_y(i  ,j  , ID);
      qright.p  = Qp_y(i  ,j  , IP);
      qright.u  = Qp_y(i  ,j  , IV); // watchout IU, IV permutation
      qright.v  = Qp_y(i  ,j  , IU); // watchout IU, IV permutation
      
      // compute hydro flux_y
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_y);
            
      //
      // update hydro array
      //
      Udata(i-1,j  , ID) += - flux_x.d*dtdx;
      Udata(i-1,j  , IP) += - flux_x.p*dtdx;
      Udata(i-1,j  , IU) += - flux_x.u*dtdx;
      Udata(i-1,j  , IV) += - flux_x.v*dtdx;

      Udata(i  ,j  , ID) +=   flux_x.d*dtdx;
      Udata(i  ,j  , IP) +=   flux_x.p*dtdx;
      Udata(i  ,j  , IU) +=   flux_x.u*dtdx;
      Udata(i  ,j  , IV) +=   flux_x.v*dtdx;

      Udata(i  ,j-1, ID) += - flux_y.d*dtdy;
      Udata(i  ,j-1, IP) += - flux_y.p*dtdy;
      Udata(i  ,j-1, IU) += - flux_y.v*dtdy; // watchout IU and IV swapped
      Udata(i  ,j-1, IV) += - flux_y.u*dtdy; // watchout IU and IV swapped

      Udata(i  ,j  , ID) +=   flux_y.d*dtdy;
      Udata(i  ,j  , IP) +=   flux_y.p*dtdy;
      Udata(i  ,j  , IU) +=   flux_y.v*dtdy; // watchout IU and IV swapped
      Udata(i  ,j  , IV) +=   flux_y.u*dtdy; // watchout IU and IV swapped
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qm_x, Qm_y, Qp_x, Qp_y;
  real_t dtdx, dtdy;
  
}; // ComputeFluxesAndUpdateFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor : public HydroBaseFunctor2D {

public:

  ComputeTraceFunctor(HydroParams params,
		      DataArray Udata,
		      DataArray Qdata,
		      DataArray Qm_x,
		      DataArray Qm_y,
		      DataArray Qp_x,
		      DataArray Qp_y,
		      real_t dtdx,
		      real_t dtdy) :
    HydroBaseFunctor2D(params),
    Udata(Udata), Qdata(Qdata),
    Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    if(j >= 1 && j <= jsize - ghostWidth &&
       i >= 1 && i <= isize - ghostWidth) {

      HydroState qLoc   ; // local primitive variables
      HydroState qPlusX ;
      HydroState qMinusX;
      HydroState qPlusY ;
      HydroState qMinusY;

      HydroState dqX;
      HydroState dqY;

      HydroState qmX;
      HydroState qmY;
      HydroState qpX;
      HydroState qpY;

      
      // get primitive variables state vector
      {
	qLoc   .d = Qdata(i  ,j  , ID);
	qPlusX .d = Qdata(i+1,j  , ID);
	qMinusX.d = Qdata(i-1,j  , ID);
	qPlusY .d = Qdata(i  ,j+1, ID);
	qMinusY.d = Qdata(i  ,j-1, ID);

	qLoc   .p = Qdata(i  ,j  , IP);
	qPlusX .p = Qdata(i+1,j  , IP);
	qMinusX.p = Qdata(i-1,j  , IP);
	qPlusY .p = Qdata(i  ,j+1, IP);
	qMinusY.p = Qdata(i  ,j-1, IP);

	qLoc   .u = Qdata(i  ,j  , IU);
	qPlusX .u = Qdata(i+1,j  , IU);
	qMinusX.u = Qdata(i-1,j  , IU);
	qPlusY .u = Qdata(i  ,j+1, IU);
	qMinusY.u = Qdata(i  ,j-1, IU);

	qLoc   .v = Qdata(i  ,j  , IV);
	qPlusX .v = Qdata(i+1,j  , IV);
	qMinusX.v = Qdata(i-1,j  , IV);
	qPlusY .v = Qdata(i  ,j+1, IV);
	qMinusY.v = Qdata(i  ,j-1, IV);

      } // 
      
      // get hydro slopes dq
      slope_unsplit_hydro_2d(&qLoc, 
			     &qPlusX, &qMinusX, 
			     &qPlusY, &qMinusY, 
			     &dqX, &dqY);
      
      // compute qm, qp
      trace_unsplit_hydro_2d(&qLoc, 
			     &dqX, &dqY,
			     dtdx, dtdy, 
			     &qmX, &qmY,
			     &qpX, &qpY);

      // store qm, qp : only what is really needed
      Qm_x(i  ,j  , ID) = qmX.d;
      Qp_x(i  ,j  , ID) = qpX.d;
      Qm_y(i  ,j  , ID) = qmY.d;
      Qp_y(i  ,j  , ID) = qpY.d;
      
      Qm_x(i  ,j  , IP) = qmX.p;
      Qp_x(i  ,j  , ID) = qpX.p;
      Qm_y(i  ,j  , ID) = qmY.p;
      Qp_y(i  ,j  , ID) = qpY.p;
      
      Qm_x(i  ,j  , IU) = qmX.u;
      Qp_x(i  ,j  , IU) = qpX.u;
      Qm_y(i  ,j  , IU) = qmY.u;
      Qp_y(i  ,j  , IU) = qpY.u;
      
      Qm_x(i  ,j  , IV) = qmX.v;
      Qp_x(i  ,j  , IV) = qpX.v;
      Qm_y(i  ,j  , IV) = qmY.v;
      Qp_y(i  ,j  , IV) = qpY.v;
      
    }
  }

  DataArray Udata, Qdata;
  DataArray Qm_x, Qm_y, Qp_x, Qp_y;
  real_t dtdx, dtdy;
  
}; // ComputeTraceFunctor


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAndStoreFluxesFunctor : public HydroBaseFunctor2D {

public:

  ComputeAndStoreFluxesFunctor(HydroParams params,
			       DataArray Qdata,
			       DataArray FluxData_x,
			       DataArray FluxData_y,		       
			       real_t dtdx,
			       real_t dtdy) :
    HydroBaseFunctor2D(params),
    Qdata(Qdata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y), 
    dtdx(dtdx),
    dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    if(j >= ghostWidth && j <= jsize-ghostWidth  &&
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
      
      // Local slopes and neighbor slopes
      HydroState dqX;
      HydroState dqY;
      HydroState dqX_neighbor;
      HydroState dqY_neighbor;

      // Local variables for Riemann problems solving
      HydroState qleft;
      HydroState qright;
      HydroState qgdnv;
      HydroState flux_x;
      HydroState flux_y;

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along X !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // get primitive variables state vector
      qLoc.d         = Qdata(i  ,j  , ID);
      qNeighbors_0.d = Qdata(i+1,j  , ID);
      qNeighbors_1.d = Qdata(i-1,j  , ID);
      qNeighbors_2.d = Qdata(i  ,j+1, ID);
      qNeighbors_3.d = Qdata(i  ,j-1, ID);
      
      qLoc.p         = Qdata(i  ,j  , IP);
      qNeighbors_0.p = Qdata(i+1,j  , IP);
      qNeighbors_1.p = Qdata(i-1,j  , IP);
      qNeighbors_2.p = Qdata(i  ,j+1, IP);
      qNeighbors_3.p = Qdata(i  ,j-1, IP);
      
      qLoc.u         = Qdata(i  ,j  , IU);
      qNeighbors_0.u = Qdata(i+1,j  , IU);
      qNeighbors_1.u = Qdata(i-1,j  , IU);
      qNeighbors_2.u = Qdata(i  ,j+1, IU);
      qNeighbors_3.u = Qdata(i  ,j-1, IU);
      
      qLoc.v         = Qdata(i  ,j  , IV);
      qNeighbors_0.v = Qdata(i+1,j  , IV);
      qNeighbors_1.v = Qdata(i-1,j  , IV);
      qNeighbors_2.v = Qdata(i  ,j+1, IV);
      qNeighbors_3.v = Qdata(i  ,j-1, IV);
      
      slope_unsplit_hydro_2d(&qLoc, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &dqX, &dqY);
	
      // slopes at left neighbor along X      
      qLocNeighbor.d = Qdata(i-1,j  , ID);
      qNeighbors_0.d = Qdata(i  ,j  , ID);
      qNeighbors_1.d = Qdata(i-2,j  , ID);
      qNeighbors_2.d = Qdata(i-1,j+1, ID);
      qNeighbors_3.d = Qdata(i-2,j-1, ID);
      
      qLocNeighbor.p = Qdata(i-1,j  , IP);
      qNeighbors_0.p = Qdata(i  ,j  , IP);
      qNeighbors_1.p = Qdata(i-2,j  , IP);
      qNeighbors_2.p = Qdata(i-1,j+1, IP);
      qNeighbors_3.p = Qdata(i-2,j-1, IP);
      
      qLocNeighbor.u = Qdata(i-1,j  , IU);
      qNeighbors_0.u = Qdata(i  ,j  , IU);
      qNeighbors_1.u = Qdata(i-2,j  , IU);
      qNeighbors_2.u = Qdata(i-1,j+1, IU);
      qNeighbors_3.u = Qdata(i-2,j-1, IU);
      
      qLocNeighbor.v = Qdata(i-1,j  , IV);
      qNeighbors_0.v = Qdata(i  ,j  , IV);
      qNeighbors_1.v = Qdata(i-2,j  , IV);
      qNeighbors_2.v = Qdata(i-1,j+1, IV);
      qNeighbors_3.v = Qdata(i-2,j-1, IV);
      
      slope_unsplit_hydro_2d(&qLocNeighbor, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &dqX_neighbor, &dqY_neighbor);
      
      //
      // compute reconstructed states at left interface along X
      //
      
      // left interface : right state
      trace_unsplit_2d_along_dir(&qLoc,
				 &dqX, &dqY,
				 dtdx, dtdy, FACE_XMIN, &qright);
      
      // left interface : left state
      trace_unsplit_2d_along_dir(&qLocNeighbor,
				 &dqX_neighbor,&dqY_neighbor,
				 dtdx, dtdy, FACE_XMAX, &qleft);
      
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //riemann_2d(qleft,qright,&qgdnv,&flux_x);
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_x);
	
      //
      // store fluxes X
      //
      FluxData_x(i  ,j , ID) = flux_x.d * dtdx;
      FluxData_x(i  ,j , IP) = flux_x.p * dtdx;
      FluxData_x(i  ,j , IU) = flux_x.u * dtdx;
      FluxData_x(i  ,j , IV) = flux_x.v * dtdx;
      
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      qLocNeighbor.d = Qdata(i  ,j-1, ID);
      qNeighbors_0.d = Qdata(i+1,j-1, ID);
      qNeighbors_1.d = Qdata(i-1,j-1, ID);
      qNeighbors_2.d = Qdata(i  ,j  , ID);
      qNeighbors_3.d = Qdata(i  ,j-2, ID);
      
      qLocNeighbor.p = Qdata(i  ,j-1, IP);
      qNeighbors_0.p = Qdata(i+1,j-1, IP);
      qNeighbors_1.p = Qdata(i-1,j-1, IP);
      qNeighbors_2.p = Qdata(i  ,j  , IP);
      qNeighbors_3.p = Qdata(i  ,j-2, IP);
      
      qLocNeighbor.u = Qdata(i  ,j-1, IU);
      qNeighbors_0.u = Qdata(i+1,j-1, IU);
      qNeighbors_1.u = Qdata(i-1,j-1, IU);
      qNeighbors_2.u = Qdata(i  ,j  , IU);
      qNeighbors_3.u = Qdata(i  ,j-2, IU);
      
      qLocNeighbor.v = Qdata(i  ,j-1, IV);
      qNeighbors_0.v = Qdata(i+1,j-1, IV);
      qNeighbors_1.v = Qdata(i-1,j-1, IV);
      qNeighbors_2.v = Qdata(i  ,j  , IV);
      qNeighbors_3.v = Qdata(i  ,j-2, IV);
	
      slope_unsplit_hydro_2d(&qLocNeighbor, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &dqX_neighbor, &dqY_neighbor);

      //
      // compute reconstructed states at left interface along Y
      //
	
      // left interface : right state
      trace_unsplit_2d_along_dir(&qLoc,
				 &dqX, &dqY,
				 dtdx, dtdy, FACE_YMIN, &qright);

      // left interface : left state
      trace_unsplit_2d_along_dir(&qLocNeighbor,
				 &dqX_neighbor,&dqY_neighbor,
				 dtdx, dtdy, FACE_YMAX, &qleft);

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swapValues(&(qleft.u) ,&(qleft.v) );
      swapValues(&(qright.u),&(qright.v));
      //riemann_2d(qleft,qright,&qgdnv,&flux_y);
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_y);

      //
      // store fluxes Y
      //
      FluxData_y(i  ,j , ID) = flux_y.d * dtdy;
      FluxData_y(i  ,j , IP) = flux_y.p * dtdy;
      FluxData_y(i  ,j , IU) = flux_y.u * dtdy;
      FluxData_y(i  ,j , IV) = flux_y.v * dtdy;
          
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray FluxData_x;
  DataArray FluxData_y;
  real_t dtdx, dtdy;
  
}; // ComputeAndStoreFluxesFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor : public HydroBaseFunctor2D {

public:

  UpdateFunctor(HydroParams params,
		DataArray Udata,
		DataArray FluxData_x,
		DataArray FluxData_y) :
    HydroBaseFunctor2D(params),
    Udata(Udata), 
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
        
    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      Udata(i  ,j  , ID) +=  FluxData_x(i  ,j  , ID);
      Udata(i  ,j  , IP) +=  FluxData_x(i  ,j  , IP);
      Udata(i  ,j  , IU) +=  FluxData_x(i  ,j  , IU);
      Udata(i  ,j  , IV) +=  FluxData_x(i  ,j  , IV);

      Udata(i  ,j  , ID) -=  FluxData_x(i+1,j  , ID);
      Udata(i  ,j  , IP) -=  FluxData_x(i+1,j  , IP);
      Udata(i  ,j  , IU) -=  FluxData_x(i+1,j  , IU);
      Udata(i  ,j  , IV) -=  FluxData_x(i+1,j  , IV);
      
      Udata(i  ,j  , ID) +=  FluxData_y(i  ,j  , ID);
      Udata(i  ,j  , IP) +=  FluxData_y(i  ,j  , IP);
      Udata(i  ,j  , IU) +=  FluxData_y(i  ,j  , IV); //
      Udata(i  ,j  , IV) +=  FluxData_y(i  ,j  , IU); //
      
      Udata(i  ,j  , ID) -=  FluxData_y(i  ,j+1, ID);
      Udata(i  ,j  , IP) -=  FluxData_y(i  ,j+1, IP);
      Udata(i  ,j  , IU) -=  FluxData_y(i  ,j+1, IV); //
      Udata(i  ,j  , IV) -=  FluxData_y(i  ,j+1, IU); //

    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData_x;
  DataArray FluxData_y;
  
}; // UpdateFunctor


/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class UpdateDirFunctor : public HydroBaseFunctor2D {

public:

  UpdateDirFunctor(HydroParams params,
		   DataArray Udata,
		   DataArray FluxData) :
    HydroBaseFunctor2D(params),
    Udata(Udata), 
    FluxData(FluxData) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      if (dir == XDIR) {

	Udata(i  ,j  , ID) +=  FluxData(i  ,j  , ID);
	Udata(i  ,j  , IP) +=  FluxData(i  ,j  , IP);
	Udata(i  ,j  , IU) +=  FluxData(i  ,j  , IU);
	Udata(i  ,j  , IV) +=  FluxData(i  ,j  , IV);

	Udata(i  ,j  , ID) -=  FluxData(i+1,j  , ID);
	Udata(i  ,j  , IP) -=  FluxData(i+1,j  , IP);
	Udata(i  ,j  , IU) -=  FluxData(i+1,j  , IU);
	Udata(i  ,j  , IV) -=  FluxData(i+1,j  , IV);

      } else if (dir == YDIR) {

	Udata(i  ,j  , ID) +=  FluxData(i  ,j  , ID);
	Udata(i  ,j  , IP) +=  FluxData(i  ,j  , IP);
	Udata(i  ,j  , IU) +=  FluxData(i  ,j  , IU);
	Udata(i  ,j  , IV) +=  FluxData(i  ,j  , IV);
	
	Udata(i  ,j  , ID) -=  FluxData(i  ,j+1, ID);
	Udata(i  ,j  , IP) -=  FluxData(i  ,j+1, IP);
	Udata(i  ,j  , IU) -=  FluxData(i  ,j+1, IU);
	Udata(i  ,j  , IV) -=  FluxData(i  ,j+1, IV);

      }
      
    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData;
  
}; // UpdateDirFunctor

    
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor : public HydroBaseFunctor2D {
  
public:
  
  ComputeSlopesFunctor(HydroParams params,
		       DataArray Qdata,
		       DataArray Slopes_x,
		       DataArray Slopes_y) :
    HydroBaseFunctor2D(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    if(j >= ghostWidth-1 && j <= jsize-ghostWidth  &&
       i >= ghostWidth-1 && i <= isize-ghostWidth ) {

      	// local primitive variables
	HydroState qLoc; // local primitive variables

	// local primitive variables in neighborbood
	HydroState qNeighbors_0;
	HydroState qNeighbors_1;
	HydroState qNeighbors_2;
	HydroState qNeighbors_3;

	// Local slopes and neighbor slopes
	HydroState dqX;
	HydroState dqY;
      
	// get primitive variables state vector
	qLoc.d         = Qdata(i  ,j  , ID);
	qNeighbors_0.d = Qdata(i+1,j  , ID);
	qNeighbors_1.d = Qdata(i-1,j  , ID);
	qNeighbors_2.d = Qdata(i  ,j+1, ID);
	qNeighbors_3.d = Qdata(i  ,j-1, ID);

	qLoc.p         = Qdata(i  ,j  , IP);
	qNeighbors_0.p = Qdata(i+1,j  , IP);
	qNeighbors_1.p = Qdata(i-1,j  , IP);
	qNeighbors_2.p = Qdata(i  ,j+1, IP);
	qNeighbors_3.p = Qdata(i  ,j-1, IP);
	
	qLoc.u         = Qdata(i  ,j  , IU);
	qNeighbors_0.u = Qdata(i+1,j  , IU);
	qNeighbors_1.u = Qdata(i-1,j  , IU);
	qNeighbors_2.u = Qdata(i  ,j+1, IU);
	qNeighbors_3.u = Qdata(i  ,j-1, IU);
	
	qLoc.v         = Qdata(i  ,j  , IV);
	qNeighbors_0.v = Qdata(i+1,j  , IV);
	qNeighbors_1.v = Qdata(i-1,j  , IV);
	qNeighbors_2.v = Qdata(i  ,j+1, IV);
	qNeighbors_3.v = Qdata(i  ,j-1, IV);
	
	slope_unsplit_hydro_2d(&qLoc, 
			       &qNeighbors_0, &qNeighbors_1, 
			       &qNeighbors_2, &qNeighbors_3,
			       &dqX, &dqY);
	
	// copy back slopes in global arrays
	Slopes_x(i  ,j , ID) = dqX.d;
	Slopes_y(i  ,j , ID) = dqY.d;
	
	Slopes_x(i  ,j , IP) = dqX.p;
	Slopes_y(i  ,j , IP) = dqY.p;
	
	Slopes_x(i  ,j , IU) = dqX.u;
	Slopes_y(i  ,j , IU) = dqY.u;
	
	Slopes_x(i  ,j , IV) = dqX.v;
	Slopes_y(i  ,j , IV) = dqY.v;
      
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray Slopes_x, Slopes_y;
  
}; // ComputeSlopesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor : public HydroBaseFunctor2D {
  
public:
  
  ComputeTraceAndFluxes_Functor(HydroParams params,
				DataArray Qdata,
				DataArray Slopes_x,
				DataArray Slopes_y,
				DataArray Fluxes,
				real_t    dtdx,
				real_t    dtdy) :
    HydroBaseFunctor2D(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y),
    Fluxes(Fluxes),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    if(j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {

	// local primitive variables
	HydroState qLoc; // local primitive variables

	// local primitive variables in neighbor cell
	HydroState qLocNeighbor;

	// Local slopes and neighbor slopes
	HydroState dqX;
	HydroState dqY;
	HydroState dqX_neighbor;
	HydroState dqY_neighbor;

	// Local variables for Riemann problems solving
	HydroState qleft;
	HydroState qright;
	HydroState qgdnv;
	HydroState flux;

	//
	// compute reconstructed states at left interface along X
	//
	qLoc.d = Qdata   (i  ,j  , ID);
	dqX.d  = Slopes_x(i  ,j  , ID);
	dqY.d  = Slopes_y(i  ,j  , ID);
	
	qLoc.p = Qdata   (i  ,j  , IP);
	dqX.p  = Slopes_x(i  ,j  , IP);
	dqY.p  = Slopes_y(i  ,j  , IP);
	
	qLoc.u = Qdata   (i  ,j  , IU);
	dqX.u  = Slopes_x(i  ,j  , IU);
	dqY.u  = Slopes_y(i  ,j  , IU);
	
	qLoc.v = Qdata   (i  ,j  , IV);
	dqX.v  = Slopes_x(i  ,j  , IV);
	dqY.v  = Slopes_y(i  ,j  , IV);

	if (dir == XDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(&qLoc,
				     &dqX, &dqY,
				     dtdx, dtdy, FACE_XMIN, &qright);
	  
	  qLocNeighbor.d = Qdata   (i-1,j  , ID);
	  dqX_neighbor.d = Slopes_x(i-1,j  , ID);
	  dqY_neighbor.d = Slopes_y(i-1,j  , ID);
	  
	  qLocNeighbor.p = Qdata   (i-1,j  , IP);
	  dqX_neighbor.p = Slopes_x(i-1,j  , IP);
	  dqY_neighbor.p = Slopes_y(i-1,j  , IP);
	  
	  qLocNeighbor.u = Qdata   (i-1,j  , IU);
	  dqX_neighbor.u = Slopes_x(i-1,j  , IU);
	  dqY_neighbor.u = Slopes_y(i-1,j  , IU);
	  
	  qLocNeighbor.v = Qdata   (i-1,j  , IV);
	  dqX_neighbor.v = Slopes_x(i-1,j  , IV);
	  dqY_neighbor.v = Slopes_y(i-1,j  , IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(&qLocNeighbor,
				     &dqX_neighbor,&dqY_neighbor,
				     dtdx, dtdy, FACE_XMAX, &qleft);
	  
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann_hllc(&qleft,&qright,&qgdnv,&flux);

	  //
	  // store fluxes
	  //	
	  Fluxes(i  ,j  , ID) =  flux.d*dtdx;
	  Fluxes(i  ,j  , IP) =  flux.p*dtdx;
	  Fluxes(i  ,j  , IU) =  flux.u*dtdx;
	  Fluxes(i  ,j  , IV) =  flux.v*dtdx;

	} else if (dir == YDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(&qLoc,
				     &dqX, &dqY,
				     dtdx, dtdy, FACE_YMIN, &qright);
	  
	  qLocNeighbor.d = Qdata   (i  ,j-1, ID);
	  dqX_neighbor.d = Slopes_x(i  ,j-1, ID);
	  dqY_neighbor.d = Slopes_y(i  ,j-1, ID);
	  
	  qLocNeighbor.p = Qdata   (i  ,j-1, IP);
	  dqX_neighbor.p = Slopes_x(i  ,j-1, IP);
	  dqY_neighbor.p = Slopes_y(i  ,j-1, IP);
	  
	  qLocNeighbor.u = Qdata   (i  ,j-1, IU);
	  dqX_neighbor.u = Slopes_x(i  ,j-1, IU);
	  dqY_neighbor.u = Slopes_y(i  ,j-1, IU);
	  
	  qLocNeighbor.v = Qdata   (i  ,j-1, IV);
	  dqX_neighbor.v = Slopes_x(i  ,j-1, IV);
	  dqY_neighbor.v = Slopes_y(i  ,j-1, IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(&qLocNeighbor,
				     &dqX_neighbor,&dqY_neighbor,
				     dtdx, dtdy, FACE_YMAX, &qleft);
	  
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  swapValues(&(qleft.u) ,&(qleft.v) );
	  swapValues(&(qright.u),&(qright.v));
	  riemann_hllc(&qleft,&qright,&qgdnv,&flux);
	  
	  //
	  // update hydro array
	  //	  
	  Fluxes(i  ,j  , ID) =  flux.d*dtdy;
	  Fluxes(i  ,j  , IP) =  flux.p*dtdy;
	  Fluxes(i  ,j  , IU) =  flux.v*dtdy; // IU/IV swapped
	  Fluxes(i  ,j  , IV) =  flux.u*dtdy; // IU/IV swapped

	}
	      
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray Slopes_x, Slopes_y;
  DataArray Fluxes;
  real_t dtdx, dtdy;
  
}; // ComputeTraceAndFluxes_Functor

/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor : public HydroBaseFunctor2D {

public:
  InitImplodeFunctor(HydroParams params,
		     DataArray Udata) :
    HydroBaseFunctor2D(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {

    const int ghostWidth = params.ghostWidth;
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
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
    
  } // end operator ()

  DataArray Udata;

}; // InitImplodeFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor : public HydroBaseFunctor2D {

public:
  InitBlastFunctor(HydroParams params,
		   DataArray Udata) :
    HydroBaseFunctor2D(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i, const int& j) const
  {

    const int ghostWidth = params.ghostWidth;
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;

    // blast problem parameters
    const real_t blast_radius      = params.blast_radius;
    const real_t radius2           = blast_radius*blast_radius;
    const real_t blast_center_x    = params.blast_center_x;
    const real_t blast_center_y    = params.blast_center_y;
    const real_t blast_density_in  = params.blast_density_in;
    const real_t blast_density_out = params.blast_density_out;
    const real_t blast_pressure_in = params.blast_pressure_in;
    const real_t blast_pressure_out= params.blast_pressure_out;
  

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
    
  } // end operator ()
  
  DataArray Udata;
  
}; // InitBlastFunctor
  

/*************************************************/
/*************************************************/
/*************************************************/
 template <FaceIdType faceId>
 class MakeBoundariesFunctor : public HydroBaseFunctor2D {

public:

  MakeBoundariesFunctor(HydroParams params,
			DataArray Udata) :
    HydroBaseFunctor2D(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int nx = params.nx;
    const int ny = params.ny;
    
    const int ghostWidth = params.ghostWidth;

    const int imin = params.imin;
    const int imax = params.imax;
    
    const int jmin = params.jmin;
    const int jmax = params.jmax;
    
    int i,j;

    int boundary_type;
    
    int i0, j0;
    int iVar;

    if (faceId == FACE_XMIN) {
      
      // boundary xmin
      boundary_type = params.boundary_type_xmin;

      j = index / ghostWidth;
      i = index - j*ghostWidth;
      
      if(j >= jmin && j <= jmax    &&
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
	  
	  Udata(i  ,j  , iVar) = Udata(i0 ,j  , iVar)*sign;
	  
	}
	
      }
    }

    if (faceId == FACE_XMAX) {
      
      // boundary xmax
      boundary_type = params.boundary_type_xmax;

      j = index / ghostWidth;
      i = index - j*ghostWidth;
      i += (nx+ghostWidth);

      if(j >= jmin          && j <= jmax             &&
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
	  
	  Udata(i  ,j  , iVar) = Udata(i0 ,j  , iVar)*sign;
	  
	}
      }
    }
    
    if (faceId == FACE_YMIN) {
      
      // boundary ymin
      boundary_type = params.boundary_type_ymin;

      i = index / ghostWidth;
      j = index - i*ghostWidth;

      if(i >= imin && i <= imax    &&
	 j >= 0    && j <ghostWidth) {
	
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
	  
	  Udata(i  ,j  , iVar) = Udata(i  ,j0  , iVar)*sign;
	}
      }
    }

    if (faceId == FACE_YMAX) {

      // boundary ymax
      boundary_type = params.boundary_type_ymax;

      i = index / ghostWidth;
      j = index - i*ghostWidth;
      j += (ny+ghostWidth);
      if(i >= imin          && i <= imax              &&
	 j >= ny+ghostWidth && j <= ny+2*ghostWidth-1) {
	
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
	  
	  Udata(i  ,j  , iVar) = Udata(i  ,j0  , iVar)*sign;
	  
	}

      }
    }
    
  } // end operator ()

  DataArray Udata;
  
}; // MakeBoundariesFunctor
  
#endif // HYDRO_RUN_FUNCTORS_H_

