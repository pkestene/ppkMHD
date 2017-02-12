#ifndef HYDRO_RUN_FUNCTORS_H_
#define HYDRO_RUN_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__


#include "HydroBaseFunctor.h"

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor : public HydroBaseFunctor {

public:
  
  ComputeDtFunctor(HydroParams params,
		   DataArray Udata) :
    HydroBaseFunctor(params),
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
      uLoc.d = Udata(index,ID);
      uLoc.p = Udata(index,IP);
      uLoc.u = Udata(index,IU);
      uLoc.v = Udata(index,IV);

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
class ConvertToPrimitivesFunctor : public HydroBaseFunctor {

public:

  ConvertToPrimitivesFunctor(HydroParams params,
			     DataArray Udata,
			     DataArray Qdata) :
    HydroBaseFunctor(params), Udata(Udata), Qdata(Qdata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    //const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= 0 && j < jsize  &&
       i >= 0 && i < isize ) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc.d = Udata(index,ID);
      uLoc.p = Udata(index,IP);
      uLoc.u = Udata(index,IU);
      uLoc.v = Udata(index,IV);
      
      // get primitive variables in current cell
      computePrimitives(&uLoc, &c, &qLoc);

      // copy q state in q global
      Qdata(index,ID) = qLoc.d;
      Qdata(index,IP) = qLoc.p;
      Qdata(index,IU) = qLoc.u;
      Qdata(index,IV) = qLoc.v;
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qdata;
    
}; // ConvertToPrimitivesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndUpdateFunctor : public HydroBaseFunctor {

public:

  ComputeFluxesAndUpdateFunctor(HydroParams params,
				DataArray Udata,
				DataArray Qm_x,
				DataArray Qm_y,
				DataArray Qp_x,
				DataArray Qp_y,
				real_t dtdx,
				real_t dtdy) :
    HydroBaseFunctor(params), Udata(Udata),
    Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index_) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index_,i,j,isize,jsize);
    
    if(j >= ghostWidth && j <= jsize - ghostWidth &&
       i >= ghostWidth && i <= isize - ghostWidth) {
      
      HydroState qleft, qright;
      HydroState flux_x, flux_y;
      HydroState qgdnv;
      int index;

      //
      // Solve Riemann problem at X-interfaces and compute
      // X-fluxes
      //
      index = coord2index(i-1,j,isize,jsize);
      qleft.d   = Qm_x(index , ID);
      qleft.p   = Qm_x(index , IP);
      qleft.u   = Qm_x(index , IU);
      qleft.v   = Qm_x(index , IV);
      
      index = coord2index(i,j,isize,jsize);
      qright.d  = Qp_x(index , ID);
      qright.p  = Qp_x(index , IP);
      qright.u  = Qp_x(index , IU);
      qright.v  = Qp_x(index , IV);
      
      // compute hydro flux_x
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_x);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      index = coord2index(i,j-1,isize,jsize);
      qleft.d   = Qm_y(index , ID);
      qleft.p   = Qm_y(index , IP);
      qleft.u   = Qm_y(index , IV); // watchout IU, IV permutation
      qleft.v   = Qm_y(index , IU); // watchout IU, IV permutation

      index = coord2index(i,j,isize,jsize);
      qright.d  = Qp_y(index , ID);
      qright.p  = Qp_y(index , IP);
      qright.u  = Qp_y(index , IV); // watchout IU, IV permutation
      qright.v  = Qp_y(index , IU); // watchout IU, IV permutation
      
      // compute hydro flux_y
      riemann_hllc(&qleft,&qright,&qgdnv,&flux_y);
            
      //
      // update hydro array
      //
      index = coord2index(i-1,j,isize,jsize);
      Udata(index , ID) += - flux_x.d*dtdx;
      Udata(index , IP) += - flux_x.p*dtdx;
      Udata(index , IU) += - flux_x.u*dtdx;
      Udata(index , IV) += - flux_x.v*dtdx;

      index = coord2index(i,j,isize,jsize);
      Udata(index , ID) +=   flux_x.d*dtdx;
      Udata(index , IP) +=   flux_x.p*dtdx;
      Udata(index , IU) +=   flux_x.u*dtdx;
      Udata(index , IV) +=   flux_x.v*dtdx;

      index = coord2index(i,j-1,isize,jsize);
      Udata(index , ID) += - flux_y.d*dtdy;
      Udata(index , IP) += - flux_y.p*dtdy;
      Udata(index , IU) += - flux_y.v*dtdy; // watchout IU and IV swapped
      Udata(index , IV) += - flux_y.u*dtdy; // watchout IU and IV swapped

      index = coord2index(i,j,isize,jsize);
      Udata(index , ID) +=   flux_y.d*dtdy;
      Udata(index , IP) +=   flux_y.p*dtdy;
      Udata(index , IU) +=   flux_y.v*dtdy; // watchout IU and IV swapped
      Udata(index , IV) +=   flux_y.u*dtdy; // watchout IU and IV swapped
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qm_x, Qm_y, Qp_x, Qp_y;
  real_t dtdx, dtdy;
  
}; // ComputeFluxesAndUpdateFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor : public HydroBaseFunctor {

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
    HydroBaseFunctor(params),
    Udata(Udata), Qdata(Qdata),
    Qm_x(Qm_x), Qm_y(Qm_y), Qp_x(Qp_x), Qp_y(Qp_y),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    int index0, index1, index2, index3;
    
    if(j >= 1 && j <= jsize - ghostWidth &&
       i >= 1 && i <= isize - ghostWidth) {

      HydroState qLoc   ; // local primitive variables
      HydroState qPlusX ;
      HydroState qMinusX;
      HydroState qPlusY ;
      HydroState qMinusY;

      HydroState dqX = {.d = 0.0, .p = 0.0, .u = 0.0, .v = 0.0};
      HydroState dqY = {.d = 0.0, .p = 0.0, .u = 0.0, .v = 0.0};

      HydroState qmX;
      HydroState qmY;
      HydroState qpX;
      HydroState qpY;

      index0 = coord2index(i+1,j  ,isize,jsize);
      index1 = coord2index(i-1,j  ,isize,jsize);
      index2 = coord2index(i  ,j+1,isize,jsize);
      index3 = coord2index(i  ,j-1,isize,jsize);
      
      // get primitive variables state vector
      {
	qLoc   .d = Qdata(index , ID);
	qPlusX .d = Qdata(index0, ID);
	qMinusX.d = Qdata(index1, ID);
	qPlusY .d = Qdata(index2, ID);
	qMinusY.d = Qdata(index3, ID);

	qLoc   .p = Qdata(index , IP);
	qPlusX .p = Qdata(index0, IP);
	qMinusX.p = Qdata(index1, IP);
	qPlusY .p = Qdata(index2, IP);
	qMinusY.p = Qdata(index3, IP);

	qLoc   .u = Qdata(index , IU);
	qPlusX .u = Qdata(index0, IU);
	qMinusX.u = Qdata(index1, IU);
	qPlusY .u = Qdata(index2, IU);
	qMinusY.u = Qdata(index3, IU);

	qLoc   .v = Qdata(index , IV);
	qPlusX .v = Qdata(index0, IV);
	qMinusX.v = Qdata(index1, IV);
	qPlusY .v = Qdata(index2, IV);
	qMinusY.v = Qdata(index3, IV);

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
      Qm_x(index, ID) = qmX.d;
      Qp_x(index, ID) = qpX.d;
      Qm_y(index, ID) = qmY.d;
      Qp_y(index, ID) = qpY.d;
      
      Qm_x(index, IP) = qmX.p;
      Qp_x(index, ID) = qpX.p;
      Qm_y(index, ID) = qmY.p;
      Qp_y(index, ID) = qpY.p;
      
      Qm_x(index, IU) = qmX.u;
      Qp_x(index, IU) = qpX.u;
      Qm_y(index, IU) = qmY.u;
      Qp_y(index, IU) = qpY.u;
      
      Qm_x(index, IV) = qmX.v;
      Qp_x(index, IV) = qpX.v;
      Qm_y(index, IV) = qmY.v;
      Qp_y(index, IV) = qpY.v;
      
    }
  }

  DataArray Udata, Qdata;
  DataArray Qm_x, Qm_y, Qp_x, Qp_y;
  real_t dtdx, dtdy;
  
}; // ComputeTraceFunctor


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeAndStoreFluxesFunctor : public HydroBaseFunctor {

public:

  ComputeAndStoreFluxesFunctor(HydroParams params,
			       DataArray Qdata,
			       DataArray FluxData_x,
			       DataArray FluxData_y,		       
			       real_t dtdx,
			       real_t dtdy) :
    HydroBaseFunctor(params),
    Qdata(Qdata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y), 
    dtdx(dtdx),
    dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    int indexc, index0, index1, index2, index3;
    
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

      index0 = coord2index(i+1,j  ,isize,jsize);
      index1 = coord2index(i-1,j  ,isize,jsize);
      index2 = coord2index(i  ,j+1,isize,jsize);
      index3 = coord2index(i  ,j-1,isize,jsize);

      // get primitive variables state vector
      qLoc.d         = Qdata(index , ID);
      qNeighbors_0.d = Qdata(index0, ID);
      qNeighbors_1.d = Qdata(index1, ID);
      qNeighbors_2.d = Qdata(index2, ID);
      qNeighbors_3.d = Qdata(index3, ID);
      
      qLoc.p         = Qdata(index , IP);
      qNeighbors_0.p = Qdata(index0, IP);
      qNeighbors_1.p = Qdata(index1, IP);
      qNeighbors_2.p = Qdata(index2, IP);
      qNeighbors_3.p = Qdata(index3, IP);
      
      qLoc.u         = Qdata(index , IU);
      qNeighbors_0.u = Qdata(index0, IU);
      qNeighbors_1.u = Qdata(index1, IU);
      qNeighbors_2.u = Qdata(index2, IU);
      qNeighbors_3.u = Qdata(index3, IU);
      
      qLoc.v         = Qdata(index , IV);
      qNeighbors_0.v = Qdata(index0, IV);
      qNeighbors_1.v = Qdata(index1, IV);
      qNeighbors_2.v = Qdata(index2, IV);
      qNeighbors_3.v = Qdata(index3, IV);
      
      slope_unsplit_hydro_2d(&qLoc, 
			     &qNeighbors_0, &qNeighbors_1, 
			     &qNeighbors_2, &qNeighbors_3,
			     &dqX, &dqY);
	
      // slopes at left neighbor along X
      indexc = coord2index(i-1,j  ,isize,jsize);
      index0 = coord2index(i  ,j  ,isize,jsize);
      index1 = coord2index(i-2,j  ,isize,jsize);
      index2 = coord2index(i-1,j+1,isize,jsize);
      index3 = coord2index(i-2,j-1,isize,jsize);
      
      qLocNeighbor.d = Qdata(indexc, ID);
      qNeighbors_0.d = Qdata(index0, ID);
      qNeighbors_1.d = Qdata(index1, ID);
      qNeighbors_2.d = Qdata(index2, ID);
      qNeighbors_3.d = Qdata(index3, ID);
      
      qLocNeighbor.p = Qdata(indexc, IP);
      qNeighbors_0.p = Qdata(index0, IP);
      qNeighbors_1.p = Qdata(index1, IP);
      qNeighbors_2.p = Qdata(index2, IP);
      qNeighbors_3.p = Qdata(index3, IP);
      
      qLocNeighbor.u = Qdata(indexc, IU);
      qNeighbors_0.u = Qdata(index0, IU);
      qNeighbors_1.u = Qdata(index1, IU);
      qNeighbors_2.u = Qdata(index2, IU);
      qNeighbors_3.u = Qdata(index3, IU);
      
      qLocNeighbor.v = Qdata(indexc, IV);
      qNeighbors_0.v = Qdata(index0, IV);
      qNeighbors_1.v = Qdata(index1, IV);
      qNeighbors_2.v = Qdata(index2, IV);
      qNeighbors_3.v = Qdata(index3, IV);
      
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
      FluxData_x(index , ID) = flux_x.d * dtdx;
      FluxData_x(index , IP) = flux_x.p * dtdx;
      FluxData_x(index , IU) = flux_x.u * dtdx;
      FluxData_x(index , IV) = flux_x.v * dtdx;
      
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // deal with left interface along Y !
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // slopes at left neighbor along Y
      indexc = coord2index(i  ,j-1,isize,jsize);
      index0 = coord2index(i+1,j-1,isize,jsize);
      index1 = coord2index(i-1,j-1,isize,jsize);
      index2 = coord2index(i  ,j  ,isize,jsize);
      index3 = coord2index(i  ,j-2,isize,jsize);

      qLocNeighbor.d = Qdata(indexc, ID);
      qNeighbors_0.d = Qdata(index0, ID);
      qNeighbors_1.d = Qdata(index1, ID);
      qNeighbors_2.d = Qdata(index2, ID);
      qNeighbors_3.d = Qdata(index3, ID);
      
      qLocNeighbor.p = Qdata(indexc, IP);
      qNeighbors_0.p = Qdata(index0, IP);
      qNeighbors_1.p = Qdata(index1, IP);
      qNeighbors_2.p = Qdata(index2, IP);
      qNeighbors_3.p = Qdata(index3, IP);
      
      qLocNeighbor.u = Qdata(indexc, IU);
      qNeighbors_0.u = Qdata(index0, IU);
      qNeighbors_1.u = Qdata(index1, IU);
      qNeighbors_2.u = Qdata(index2, IU);
      qNeighbors_3.u = Qdata(index3, IU);
      
      qLocNeighbor.v = Qdata(indexc, IV);
      qNeighbors_0.v = Qdata(index0, IV);
      qNeighbors_1.v = Qdata(index1, IV);
      qNeighbors_2.v = Qdata(index2, IV);
      qNeighbors_3.v = Qdata(index3, IV);
	
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
      FluxData_y(index , ID) = flux_y.d * dtdy;
      FluxData_y(index , IP) = flux_y.p * dtdy;
      FluxData_y(index , IU) = flux_y.u * dtdy;
      FluxData_y(index , IV) = flux_y.v * dtdy;
          
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
class UpdateFunctor : public HydroBaseFunctor {

public:

  UpdateFunctor(HydroParams params,
		DataArray Udata,
		DataArray FluxData_x,
		DataArray FluxData_y) :
    HydroBaseFunctor(params),
    Udata(Udata), 
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    int index2;
    
    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      Udata(index , ID) +=  FluxData_x(index , ID);
      Udata(index , IP) +=  FluxData_x(index , IP);
      Udata(index , IU) +=  FluxData_x(index , IU);
      Udata(index , IV) +=  FluxData_x(index , IV);

      index2 = coord2index(i+1,j,isize,jsize);
      Udata(index , ID) -=  FluxData_x(index2 , ID);
      Udata(index , IP) -=  FluxData_x(index2 , IP);
      Udata(index , IU) -=  FluxData_x(index2 , IU);
      Udata(index , IV) -=  FluxData_x(index2 , IV);
      
      Udata(index , ID) +=  FluxData_y(index , ID);
      Udata(index , IP) +=  FluxData_y(index , IP);
      Udata(index , IU) +=  FluxData_y(index , IV); //
      Udata(index , IV) +=  FluxData_y(index , IU); //
      
      index2 = coord2index(i,j+1,isize,jsize);
      Udata(index , ID) -=  FluxData_y(index2 , ID);
      Udata(index , IP) -=  FluxData_y(index2 , IP);
      Udata(index , IU) -=  FluxData_y(index2 , IV); //
      Udata(index , IV) -=  FluxData_y(index2 , IU); //

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
class UpdateDirFunctor : public HydroBaseFunctor {

public:

  UpdateDirFunctor(HydroParams params,
		   DataArray Udata,
		   DataArray FluxData) :
    HydroBaseFunctor(params),
    Udata(Udata), 
    FluxData(FluxData) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j, index2;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      if (dir == XDIR) {

	Udata(index , ID) +=  FluxData(index , ID);
	Udata(index , IP) +=  FluxData(index , IP);
	Udata(index , IU) +=  FluxData(index , IU);
	Udata(index , IV) +=  FluxData(index , IV);

	index2 = coord2index(i+1,j,isize,jsize);
	Udata(index , ID) -=  FluxData(index2 , ID);
	Udata(index , IP) -=  FluxData(index2 , IP);
	Udata(index , IU) -=  FluxData(index2 , IU);
	Udata(index , IV) -=  FluxData(index2 , IV);

      } else if (dir == YDIR) {

	Udata(index , ID) +=  FluxData(index , ID);
	Udata(index , IP) +=  FluxData(index , IP);
	Udata(index , IU) +=  FluxData(index , IU);
	Udata(index , IV) +=  FluxData(index , IV);
	
	index2 = coord2index(i,j+1,isize,jsize);
	Udata(index , ID) -=  FluxData(index2 , ID);
	Udata(index , IP) -=  FluxData(index2 , IP);
	Udata(index , IU) -=  FluxData(index2 , IU);
	Udata(index , IV) -=  FluxData(index2 , IV);

      }
      
    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData;
  
}; // UpdateDirFunctor

    
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeSlopesFunctor : public HydroBaseFunctor {
  
public:
  
  ComputeSlopesFunctor(HydroParams params,
		       DataArray Qdata,
		       DataArray Slopes_x,
		       DataArray Slopes_y) :
    HydroBaseFunctor(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    int i,j;
    index2coord(index,i,j,isize,jsize);

    int index0, index1, index2, index3;
    
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

	index0 = coord2index(i+1,j  ,isize,jsize);
	index1 = coord2index(i-1,j  ,isize,jsize);
	index2 = coord2index(i  ,j+1,isize,jsize);
	index3 = coord2index(i  ,j-1,isize,jsize);
      
	// get primitive variables state vector
	qLoc.d         = Qdata(index  , ID);
	qNeighbors_0.d = Qdata(index0 , ID);
	qNeighbors_1.d = Qdata(index1 , ID);
	qNeighbors_2.d = Qdata(index2 , ID);
	qNeighbors_3.d = Qdata(index3 , ID);

	qLoc.p         = Qdata(index  , IP);
	qNeighbors_0.p = Qdata(index0 , IP);
	qNeighbors_1.p = Qdata(index1 , IP);
	qNeighbors_2.p = Qdata(index2 , IP);
	qNeighbors_3.p = Qdata(index3 , IP);
	
	qLoc.u         = Qdata(index  , IU);
	qNeighbors_0.u = Qdata(index0 , IU);
	qNeighbors_1.u = Qdata(index1 , IU);
	qNeighbors_2.u = Qdata(index2 , IU);
	qNeighbors_3.u = Qdata(index3 , IU);
	
	qLoc.v         = Qdata(index  , IV);
	qNeighbors_0.v = Qdata(index0 , IV);
	qNeighbors_1.v = Qdata(index1 , IV);
	qNeighbors_2.v = Qdata(index2 , IV);
	qNeighbors_3.v = Qdata(index3 , IV);
	
	slope_unsplit_hydro_2d(&qLoc, 
			       &qNeighbors_0, &qNeighbors_1, 
			       &qNeighbors_2, &qNeighbors_3,
			       &dqX, &dqY);
	
	// copy back slopes in global arrays
	Slopes_x(index, ID) = dqX.d;
	Slopes_y(index, ID) = dqY.d;
	
	Slopes_x(index, IP) = dqX.p;
	Slopes_y(index, IP) = dqY.p;
	
	Slopes_x(index, IU) = dqX.u;
	Slopes_y(index, IU) = dqY.u;
	
	Slopes_x(index, IV) = dqX.v;
	Slopes_y(index, IV) = dqY.v;
      
    } // end if
    
  } // end operator ()
  
  DataArray Qdata;
  DataArray Slopes_x, Slopes_y;
  
}; // ComputeSlopesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor : public HydroBaseFunctor {
  
public:
  
  ComputeTraceAndFluxes_Functor(HydroParams params,
				DataArray Qdata,
				DataArray Slopes_x,
				DataArray Slopes_y,
				DataArray Fluxes,
				real_t    dtdx,
				real_t    dtdy) :
    HydroBaseFunctor(params), Qdata(Qdata),
    Slopes_x(Slopes_x), Slopes_y(Slopes_y),
    Fluxes(Fluxes),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    int index2;
    
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
	qLoc.d = Qdata   (index, ID);
	dqX.d  = Slopes_x(index, ID);
	dqY.d  = Slopes_y(index, ID);
	
	qLoc.p = Qdata   (index, IP);
	dqX.p  = Slopes_x(index, IP);
	dqY.p  = Slopes_y(index, IP);
	
	qLoc.u = Qdata   (index, IU);
	dqX.u  = Slopes_x(index, IU);
	dqY.u  = Slopes_y(index, IU);
	
	qLoc.v = Qdata   (index, IV);
	dqX.v  = Slopes_x(index, IV);
	dqY.v  = Slopes_y(index, IV);

	if (dir == XDIR) {

	  index2 = coord2index(i-1,j,isize,jsize);

	  // left interface : right state
	  trace_unsplit_2d_along_dir(&qLoc,
				     &dqX, &dqY,
				     dtdx, dtdy, FACE_XMIN, &qright);
	  
	  qLocNeighbor.d = Qdata   (index2, ID);
	  dqX_neighbor.d = Slopes_x(index2, ID);
	  dqY_neighbor.d = Slopes_y(index2, ID);
	  
	  qLocNeighbor.p = Qdata   (index2, IP);
	  dqX_neighbor.p = Slopes_x(index2, IP);
	  dqY_neighbor.p = Slopes_y(index2, IP);
	  
	  qLocNeighbor.u = Qdata   (index2, IU);
	  dqX_neighbor.u = Slopes_x(index2, IU);
	  dqY_neighbor.u = Slopes_y(index2, IU);
	  
	  qLocNeighbor.v = Qdata   (index2, IV);
	  dqX_neighbor.v = Slopes_x(index2, IV);
	  dqY_neighbor.v = Slopes_y(index2, IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(&qLocNeighbor,
				     &dqX_neighbor,&dqY_neighbor,
				     dtdx, dtdy, FACE_XMAX, &qleft);
	  
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann_hllc(&qleft,&qright,&qgdnv,&flux);

	  //
	  // store fluxes
	  //	
	  Fluxes(index , ID) =  flux.d*dtdx;
	  Fluxes(index , IP) =  flux.p*dtdx;
	  Fluxes(index , IU) =  flux.u*dtdx;
	  Fluxes(index , IV) =  flux.v*dtdx;

	} else if (dir == YDIR) {

	  index2 = coord2index(i,j-1,isize,jsize);

	  // left interface : right state
	  trace_unsplit_2d_along_dir(&qLoc,
				     &dqX, &dqY,
				     dtdx, dtdy, FACE_YMIN, &qright);
	  
	  qLocNeighbor.d = Qdata   (index2, ID);
	  dqX_neighbor.d = Slopes_x(index2, ID);
	  dqY_neighbor.d = Slopes_y(index2, ID);
	  
	  qLocNeighbor.p = Qdata   (index2, IP);
	  dqX_neighbor.p = Slopes_x(index2, IP);
	  dqY_neighbor.p = Slopes_y(index2, IP);
	  
	  qLocNeighbor.u = Qdata   (index2, IU);
	  dqX_neighbor.u = Slopes_x(index2, IU);
	  dqY_neighbor.u = Slopes_y(index2, IU);
	  
	  qLocNeighbor.v = Qdata   (index2, IV);
	  dqX_neighbor.v = Slopes_x(index2, IV);
	  dqY_neighbor.v = Slopes_y(index2, IV);
	  
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
	  Fluxes(index , ID) =  flux.d*dtdy;
	  Fluxes(index , IP) =  flux.p*dtdy;
	  Fluxes(index , IU) =  flux.v*dtdy; // IU/IV swapped
	  Fluxes(index , IV) =  flux.u*dtdy; // IU/IV swapped

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
// class ComputeTraceAndUpdate_Y_Functor : public HydroBaseFunctor {
  
// public:
  
//   ComputeTraceAndUpdate_Y_Functor(HydroParams params,
// 				  DataArray Udata,
// 				  DataArray Qdata,
// 				  DataArray Slopes_x,
// 				  DataArray Slopes_y,
// 				  real_t    dtdx,
// 				  real_t    dtdy) :
//     HydroBaseFunctor(params), Udata(Udata), Qdata(Qdata),
//     Slopes_x(Slopes_x), Slopes_y(Slopes_y),
//     dtdx(dtdx), dtdy(dtdy) {};
  
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const int& index) const
//   {
//     const int isize = params.isize;
//     const int jsize = params.jsize;
//     const int ghostWidth = params.ghostWidth;
    
//     const int j = index / isize;
//     const int i = index - j*isize;
    
//     if(j >= ghostWidth && j <= jsize-ghostWidth  &&
//        i >= ghostWidth && i <= isize-ghostWidth ) {
      
//       // local primitive variables
//       HydroState qLoc; // local primitive variables
      
//       // local primitive variables in neighbor cell
//       HydroState qLocNeighbor;
      
//       // Local slopes and neighbor slopes
//       HydroState dqX;
//       HydroState dqY;
//       HydroState dqX_neighbor;
//       HydroState dqY_neighbor;
      
//       // Local variables for Riemann problems solving
//       HydroState qleft;
//       HydroState qright;
//       HydroState qgdnv;
//       HydroState flux_y;
      
//       //int index = i  + isize * j;	
      
//       //
//       // compute reconstructed states at left interface along Y
//       //
//       qLoc.d = Qdata   (index, ID);
//       dqX.d  = Slopes_x(index, ID);
//       dqY.d  = Slopes_y(index, ID);
      
//       qLoc.p = Qdata   (index, IP);
//       dqX.p  = Slopes_x(index, IP);
//       dqY.p  = Slopes_y(index, IP);
      
//       qLoc.u = Qdata   (index, IU);
//       dqX.u  = Slopes_x(index, IU);
//       dqY.u  = Slopes_y(index, IU);
      
//       qLoc.v = Qdata   (index, IV);
//       dqX.v  = Slopes_x(index, IV);
//       dqY.v  = Slopes_y(index, IV);
      
      
//     }

//   } // end operator ()
  
//   DataArray Udata, Qdata;
//   DataArray Slopes_x, Slopes_y;
//   real_t dtdx, dtdy;
  
// }; // ComputeTraceAndUpdate_Y_Functor
    
/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor : public HydroBaseFunctor {

public:
  InitImplodeFunctor(HydroParams params,
		     DataArray Udata) :
    HydroBaseFunctor(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;
    
    real_t tmp = x+y*y;
    if (tmp > 0.5 && tmp < 1.5) {
      Udata(index , ID) = 1.0;
      Udata(index , IP) = 1.0/(gamma0-1.0);
      Udata(index , IU) = 0.0;
      Udata(index , IV) = 0.0;
    } else {
      Udata(index , ID) = 0.125;
      Udata(index , IP) = 0.14/(gamma0-1.0);
      Udata(index , IU) = 0.0;
      Udata(index , IV) = 0.0;
    }
    
  } // end operator ()

  DataArray Udata;

}; // InitImplodeFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor : public HydroBaseFunctor {

public:
  InitBlastFunctor(HydroParams params,
		   DataArray Udata) :
    HydroBaseFunctor(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
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
  

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    real_t x = xmin + dx/2 + (i-ghostWidth)*dx;
    real_t y = ymin + dy/2 + (j-ghostWidth)*dy;

    real_t d2 = 
      (x-blast_center_x)*(x-blast_center_x)+
      (y-blast_center_y)*(y-blast_center_y);    
    
    if (d2 < radius2) {
      Udata(index , ID) = blast_density_in;
      Udata(index , IP) = blast_pressure_in/(gamma0-1.0);
      Udata(index , IU) = 0.0;
      Udata(index , IV) = 0.0;
    } else {
      Udata(index , ID) = blast_density_out;
      Udata(index , IP) = blast_pressure_out/(gamma0-1.0);
      Udata(index , IU) = 0.0;
      Udata(index , IV) = 0.0;
    }
    
  } // end operator ()
  
  DataArray Udata;
  
}; // InitBlastFunctor
  

/*************************************************/
/*************************************************/
/*************************************************/
 template <FaceIdType faceId>
 class MakeBoundariesFunctor : public HydroBaseFunctor {

public:

  MakeBoundariesFunctor(HydroParams params,
			DataArray Udata) :
    HydroBaseFunctor(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int nx = params.nx;
    const int ny = params.ny;
    
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    const int imin = params.imin;
    const int imax = params.imax;
    
    const int jmin = params.jmin;
    const int jmax = params.jmax;
    
    int i,j;

    int boundary_type;
    
    int i0, j0;
    int iVar;
    int index_out, index_in;

    if (faceId == FACE_XMIN) {
      
      // boundary xmin
      boundary_type = params.boundary_type_xmin;

      j = index / ghostWidth;
      i = index - j*ghostWidth;
      
      if(j >= jmin && j <= jmax    &&
	 i >= 0    && i <ghostWidth) {
	
	real_t sign=1.0;
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*ghostWidth-1-i;
	    if (iVar==IU) sign=-ONE_F;
	  } else if( boundary_type == BC_NEUMANN ) {
	    i0=ghostWidth;
	  } else { // periodic
	    i0=nx+i;
	  }
	  
	  index_out = coord2index(i ,j,isize,jsize);
	  index_in  = coord2index(i0,j,isize,jsize);
	  Udata(index_out , iVar) = Udata(index_in , iVar)*sign;
	  
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
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    i0=2*nx+2*ghostWidth-1-i;
	    if (iVar==IU) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    i0=nx+ghostWidth-1;
	  } else { // periodic
	    i0=i-nx;
	  }
	  
	  index_out = coord2index(i ,j,isize,jsize);
	  index_in  = coord2index(i0,j,isize,jsize);
	  Udata(index_out , iVar) = Udata(index_in , iVar)*sign;
	  
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
	
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ghostWidth-1-j;
	    if (iVar==IV) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ghostWidth;
	  } else { // periodic
	    j0=ny+j;
	  }
	  
	  index_out = coord2index(i,j ,isize,jsize);
	  index_in  = coord2index(i,j0,isize,jsize);
	  Udata(index_out , iVar) = Udata(index_in , iVar)*sign;
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
	for ( iVar=0; iVar<NBVAR; iVar++ ) {
	  
	  if ( boundary_type == BC_DIRICHLET ) {
	    j0=2*ny+2*ghostWidth-1-j;
	    if (iVar==IV) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ny+ghostWidth-1;
	  } else { // periodic
	    j0=j-ny;
	  }
	  
	  index_out = coord2index(i,j ,isize,jsize);
	  index_in  = coord2index(i,j0,isize,jsize);
	  Udata(index_out , iVar) = Udata(index_in , iVar)*sign;
	  
	}

      }
    }
    
  } // end operator ()

  DataArray Udata;
  
}; // MakeBoundariesFunctor
  
#endif // HYDRO_RUN_FUNCTORS_H_

