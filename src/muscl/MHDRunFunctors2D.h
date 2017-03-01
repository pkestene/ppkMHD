#ifndef MHD_RUN_FUNCTORS_2D_H_
#define MHD_RUN_FUNCTORS_2D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "kokkos_shared.h"
#include "MHDBaseFunctor2D.h"
#include "RiemannSolvers_MHD.h"

// init conditions
#include "BlastParams.h"

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

namespace ppkMHD { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor2D_MHD : public MHDBaseFunctor2D {

public:
  
  ComputeDtFunctor2D_MHD(HydroParams params,
			 DataArray2d Qdata) :
    MHDBaseFunctor2D(params),
    Qdata(Qdata)  {};

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
      
      MHDState qLoc; // primitive    variables in current cell
      
      // get primitive variables in current cell
      qLoc[ID]  = Qdata(i,j,ID);
      qLoc[IP]  = Qdata(i,j,IP);
      qLoc[IU]  = Qdata(i,j,IU);
      qLoc[IV]  = Qdata(i,j,IV);
      qLoc[IW]  = Qdata(i,j,IW);
      qLoc[IBX] = Qdata(i,j,IBX);
      qLoc[IBY] = Qdata(i,j,IBY);
      qLoc[IBZ] = Qdata(i,j,IBZ);

      // compute fastest information speeds
      real_t fastInfoSpeed[3];
      find_speed_info<TWO_D>(qLoc, fastInfoSpeed, params);
      
      real_t vx = fastInfoSpeed[IX];
      real_t vy = fastInfoSpeed[IY];
      
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

  DataArray2d Qdata;
  
}; // ComputeDtFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor2D_MHD : public MHDBaseFunctor2D {

public:

  ConvertToPrimitivesFunctor2D_MHD(HydroParams params,
				   DataArray2d Udata,
				   DataArray2d Qdata) :
    MHDBaseFunctor2D(params), Udata(Udata), Qdata(Qdata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    //const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // magnetic field in neighbor cells
    real_t magFieldNeighbors[3];
    
    if(j >= 0 && j < jsize-1  &&
       i >= 0 && i < isize-1 ) {
      
      MHDState uLoc; // conservative    variables in current cell
      MHDState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc[ID]  = Udata(i,j,ID);
      uLoc[IP]  = Udata(i,j,IP);
      uLoc[IU]  = Udata(i,j,IU);
      uLoc[IV]  = Udata(i,j,IV);
      uLoc[IW]  = Udata(i,j,IW);
      uLoc[IBX] = Udata(i,j,IBX);
      uLoc[IBY] = Udata(i,j,IBY);
      uLoc[IBZ] = Udata(i,j,IBZ);

      // get mag field in neighbor cells
      magFieldNeighbors[IX] = Udata(i+1,j  ,IBX);
      magFieldNeighbors[IY] = Udata(i  ,j+1,IBY);
      magFieldNeighbors[IZ] = 0.0;
      
      // get primitive variables in current cell
      constoprim_mhd(uLoc, magFieldNeighbors, c, qLoc);

      // copy q state in q global
      Qdata(i,j,ID)  = qLoc[ID];
      Qdata(i,j,IP)  = qLoc[IP];
      Qdata(i,j,IU)  = qLoc[IU];
      Qdata(i,j,IV)  = qLoc[IV];
      Qdata(i,j,IW)  = qLoc[IW];
      Qdata(i,j,IBX) = qLoc[IBX];
      Qdata(i,j,IBY) = qLoc[IBY];
      Qdata(i,j,IBZ) = qLoc[IBZ];
      
    }
    
  }
  
  DataArray2d Udata;
  DataArray2d Qdata;
    
}; // ConvertToPrimitivesFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndStoreFunctor2D_MHD : public MHDBaseFunctor2D {

public:

  ComputeFluxesAndStoreFunctor2D_MHD(HydroParams params,
				     DataArray2d Qm_x,
				     DataArray2d Qm_y,
				     DataArray2d Qp_x,
				     DataArray2d Qp_y,
				     DataArray2d Fluxes_x,
				     DataArray2d Fluxes_y,
				     real_t dtdx,
				     real_t dtdy) :
    MHDBaseFunctor2D(params),
    Qm_x(Qm_x), Qm_y(Qm_y),
    Qp_x(Qp_x), Qp_y(Qp_y),
    Fluxes_x(Fluxes_x), Fluxes_y(Fluxes_y),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {
      
      MHDState qleft, qright;
      MHDState flux;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //
      get_state(Qm_x, i-1, j  , qleft);
      get_state(Qp_x, i  , j  , qright);
      
      // compute hydro flux along X
      riemann_hlld(qleft,qright,flux,params);

      // store fluxes
      set_state(Fluxes_x, i  , j  , flux);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      get_state(Qm_y, i  ,j-1, qleft);
      swapValues(&(qleft[IU]) ,&(qleft[IV]) );
      swapValues(&(qleft[IBX]) ,&(qleft[IBY]) );

      get_state(Qp_y, i  ,j  , qright);
      swapValues(&(qright[IU]) ,&(qright[IV]) );
      swapValues(&(qright[IBX]) ,&(qright[IBY]) );
      
      // compute hydro flux along Y
      riemann_hlld(qleft,qright,flux,params);
            
      // store fluxes
      set_state(Fluxes_y, i  ,j  , flux);
      
    }
    
  }
  
  DataArray2d Qm_x, Qm_y, Qp_x, Qp_y;
  DataArray2d Fluxes_x, Fluxes_y;
  real_t dtdx, dtdy;
  
}; // ComputeFluxesAndStoreFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndStoreFunctor2D : public MHDBaseFunctor2D {

public:

  ComputeEmfAndStoreFunctor2D(HydroParams params,
			      DataArray2d QEdge_RT,
			      DataArray2d QEdge_RB,
			      DataArray2d QEdge_LT,
			      DataArray2d QEdge_LB,
			      DataArrayScalar Emf,
			      real_t dtdx,
			      real_t dtdy) :
    MHDBaseFunctor2D(params),
    QEdge_RT(QEdge_RT), QEdge_RB(QEdge_RB),
    QEdge_LT(QEdge_LT), QEdge_LB(QEdge_LB),
    Emf(Emf),
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {

      // in 2D, we only need to compute emfZ
      MHDState qEdge_emfZ[4];

      // preparation for calling compute_emf (equivalent to cmp_mag_flx
      // in DUMSES)
      // in the following, the 2 first indexes in qEdge_emf array play
      // the same offset role as in the calling argument of cmp_mag_flx 
      // in DUMSES (if you see what I mean ?!)
      get_state(QEdge_RT, i-1,j-1, qEdge_emfZ[IRT]);
      get_state(QEdge_RB, i-1,j  , qEdge_emfZ[IRB]);
      get_state(QEdge_LT, i  ,j-1, qEdge_emfZ[ILT]);
      get_state(QEdge_LB, i  ,j  , qEdge_emfZ[ILB]);

      // actually compute emfZ
      real_t emfZ = compute_emf<EMFZ>(qEdge_emfZ,params);
      Emf(i,j) = emfZ;
      
    }
  }

  DataArray2d QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB;
  DataArrayScalar Emf;
  real_t dtdx, dtdy;

}; // ComputeEmfAndStoreFunctor2D


/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor2D_MHD : public MHDBaseFunctor2D {

public:

  ComputeTraceFunctor2D_MHD(HydroParams params,
			    DataArray2d Udata,
			    DataArray2d Qdata,
			    DataArray2d Qm_x,
			    DataArray2d Qm_y,
			    DataArray2d Qp_x,
			    DataArray2d Qp_y,
			    DataArray2d QEdge_RT,
			    DataArray2d QEdge_RB,
			    DataArray2d QEdge_LT,
			    DataArray2d QEdge_LB,
			    real_t dtdx,
			    real_t dtdy) :
    MHDBaseFunctor2D(params),
    Udata(Udata), Qdata(Qdata),
    Qm_x(Qm_x), Qm_y(Qm_y),
    Qp_x(Qp_x), Qp_y(Qp_y),
    QEdge_RT(QEdge_RT), QEdge_RB(QEdge_RB),
    QEdge_LT(QEdge_LT), QEdge_LB(QEdge_LB), 
    dtdx(dtdx), dtdy(dtdy) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    if(j >= ghostWidth-2 && j < jsize - ghostWidth+1 &&
       i >= ghostWidth-2 && i < isize - ghostWidth+1) {

      MHDState qNb[3][3];
      BField  bfNb[4][4];
      
      MHDState qm[2];
      MHDState qp[2];

      MHDState qEdge[4];
      real_t c = 0.0;
      
      // prepare qNb : q state in the 3-by-3 neighborhood
      // note that current cell (ii,jj) is in qNb[1][1]
      // also note that the effective stencil is 4-by-4 since
      // computation of primitive variable (q) requires mag
      // field on the right (see computePrimitives_MHD_2D)
      for (int di=0; di<3; di++)
	for (int dj=0; dj<3; dj++) {
	  get_state(Qdata, i+di-1, j+dj-1, qNb[di][dj]);
	}
      
      // prepare bfNb : bf (face centered mag field) in the
      // 4-by-4 neighborhood
      // note that current cell (ii,jj) is in bfNb[1][1]
      for (int di=0; di<4; di++)
	for (int dj=0; dj<4; dj++) {
	  get_magField(Udata, i+di-1, j+dj-1, bfNb[di][dj]);
	}

      trace_unsplit_mhd_2d(qNb, bfNb, c, dtdx, dtdy, 0.0, qm, qp, qEdge);

      // store qm, qp : only what is really needed
      set_state(Qm_x, i,j, qm[0]);
      set_state(Qp_x, i,j, qp[0]);
      set_state(Qm_y, i,j, qm[1]);
      set_state(Qp_y, i,j, qp[1]);

      set_state(QEdge_RT, i,j, qEdge[IRT]);
      set_state(QEdge_RB, i,j, qEdge[IRB]);
      set_state(QEdge_LT, i,j, qEdge[ILT]);
      set_state(QEdge_LB, i,j, qEdge[ILB]);
      
    }
  }

  DataArray2d Udata, Qdata;
  DataArray2d Qm_x, Qm_y, Qp_x, Qp_y;
  DataArray2d QEdge_RT, QEdge_RB, QEdge_LT, QEdge_LB;
  real_t dtdx, dtdy;
  
}; // ComputeTraceFunctor2D_MHD

  
/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor2D_MHD : public MHDBaseFunctor2D {

public:

  UpdateFunctor2D_MHD(HydroParams params,
		      DataArray2d Udata,
		      DataArray2d FluxData_x,
		      DataArray2d FluxData_y,
		      real_t dtdx,
		      real_t dtdy) :
    MHDBaseFunctor2D(params),
    Udata(Udata), 
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

    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      MHDState udata;
      MHDState flux;
      get_state(Udata, i,j, udata);

      // add up contributions from all 4 faces
      
      get_state(FluxData_x, i,j, flux);      
      udata[ID]  +=  flux[ID]*dtdx;
      udata[IP]  +=  flux[IP]*dtdx;
      udata[IU]  +=  flux[IU]*dtdx;
      udata[IV]  +=  flux[IV]*dtdx;
      udata[IW]  +=  flux[IW]*dtdx;
      //udata[IBX] +=  flux[IBX]*dtdx;
      //udata[IBY] +=  flux[IBY]*dtdx;
      udata[IBZ] +=  flux[IBZ]*dtdx;
      
      get_state(FluxData_x, i+1,j  , flux);      
      udata[ID]  -=  flux[ID]*dtdx;
      udata[IP]  -=  flux[IP]*dtdx;
      udata[IU]  -=  flux[IU]*dtdx;
      udata[IV]  -=  flux[IV]*dtdx;
      udata[IW]  -=  flux[IW]*dtdx;
      //udata[IBX] -=  flux[IBX]*dtdx;
      //udata[IBY] -=  flux[IBY]*dtdx;
      udata[IBZ] -=  flux[IBZ]*dtdx;
      
      get_state(FluxData_y, i,j, flux);      
      udata[ID]  +=  flux[ID]*dtdy;
      udata[IP]  +=  flux[IP]*dtdy;
      udata[IU]  +=  flux[IV]*dtdy; //
      udata[IV]  +=  flux[IU]*dtdy; //
      udata[IW]  +=  flux[IW]*dtdy;
      //udata[IBX] +=  flux[IBX]*dtdy;
      //udata[IBY] +=  flux[IBY]*dtdy;
      udata[IBZ] +=  flux[IBZ]*dtdy;
                  
      get_state(FluxData_y, i,j+1, flux);
      udata[ID]  -=  flux[ID]*dtdy;
      udata[IP]  -=  flux[IP]*dtdy;
      udata[IU]  -=  flux[IV]*dtdy; //
      udata[IV]  -=  flux[IU]*dtdy; //
      udata[IW]  -=  flux[IW]*dtdy;
      //udata[IBX] -=  flux[IBX]*dtdy;
      //udata[IBY] -=  flux[IBY]*dtdy;
      udata[IBZ] -=  flux[IBZ]*dtdy;

      // write back result in Udata
      set_state(Udata, i,j, udata);
      
    } // end if
    
  } // end operator ()
  
  DataArray2d Udata;
  DataArray2d FluxData_x;
  DataArray2d FluxData_y;
  real_t dtdx, dtdy;
  
}; // UpdateFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateEmfFunctor2D : public MHDBaseFunctor2D {

public:

  UpdateEmfFunctor2D(HydroParams params,
		     DataArray2d Udata,
		     DataArrayScalar Emf,
		     real_t dtdx,
		     real_t dtdy) :
    MHDBaseFunctor2D(params),
    Udata(Udata), 
    Emf(Emf),
    dtdx(dtdx),
    dtdy(dtdy){};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize-ghostWidth /*+1*/  &&
       i >= ghostWidth && i < isize-ghostWidth /*+1*/ ) {

      //MHDState udata;
      //get_state(Udata, index, udata);

      // left-face B-field
      Udata(i,j,IA) += ( Emf(i  ,j+1) - Emf(i,j) )*dtdy;
      Udata(i,j,IB) -= ( Emf(i+1,j  ) - Emf(i,j) )*dtdx;		    

    }
  }

  DataArray2d Udata;
  DataArrayScalar Emf;
  real_t dtdx, dtdy;

}; // UpdateEmfFunctor2D
  

/*************************************************/
/*************************************************/
/*************************************************/
template <Direction dir>
class ComputeTraceAndFluxes_Functor2D_MHD : public MHDBaseFunctor2D {
  
public:
  
  ComputeTraceAndFluxes_Functor2D_MHD(HydroParams params,
				      DataArray2d Qdata,
				      DataArray2d Slopes_x,
				      DataArray2d Slopes_y,
				      DataArray2d Fluxes,
				      real_t    dtdx,
				      real_t    dtdy) :
    MHDBaseFunctor2D(params), Qdata(Qdata),
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
    
    if(j >= ghostWidth && j <= jsize-ghostWidth  &&
       i >= ghostWidth && i <= isize-ghostWidth ) {

	// local primitive variables
	MHDState qLoc; // local primitive variables

	// local primitive variables in neighbor cell
	MHDState qLocNeighbor;

	// Local slopes and neighbor slopes
	MHDState dqX;
	MHDState dqY;
	MHDState dqX_neighbor;
	MHDState dqY_neighbor;

	// Local variables for Riemann problems solving
	MHDState qleft;
	MHDState qright;
	//MHDState qgdnv;
	MHDState flux;

	//
	// compute reconstructed states at left interface along X
	//
	qLoc[ID] = Qdata   (i,j, ID);
	dqX[ID]  = Slopes_x(i,j, ID);
	dqY[ID]  = Slopes_y(i,j, ID);
	
	qLoc[IP] = Qdata   (i,j, IP);
	dqX[IP]  = Slopes_x(i,j, IP);
	dqY[IP]  = Slopes_y(i,j, IP);
	
	qLoc[IU] = Qdata   (i,j, IU);
	dqX[IU]  = Slopes_x(i,j, IU);
	dqY[IU]  = Slopes_y(i,j, IU);
	
	qLoc[IV] = Qdata   (i,j, IV);
	dqX[IV]  = Slopes_x(i,j, IV);
	dqY[IV]  = Slopes_y(i,j, IV);

	if (dir == XDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(qLoc,
				     dqX, dqY,
				     dtdx, dtdy, FACE_XMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (i-1,j, ID);
	  dqX_neighbor[ID] = Slopes_x(i-1,j, ID);
	  dqY_neighbor[ID] = Slopes_y(i-1,j, ID);
	  
	  qLocNeighbor[IP] = Qdata   (i-1,j, IP);
	  dqX_neighbor[IP] = Slopes_x(i-1,j, IP);
	  dqY_neighbor[IP] = Slopes_y(i-1,j, IP);
	  
	  qLocNeighbor[IU] = Qdata   (i-1,j, IU);
	  dqX_neighbor[IU] = Slopes_x(i-1,j, IU);
	  dqY_neighbor[IU] = Slopes_y(i-1,j, IU);
	  
	  qLocNeighbor[IV] = Qdata   (i-1,j, IV);
	  dqX_neighbor[IV] = Slopes_x(i-1,j, IV);
	  dqY_neighbor[IV] = Slopes_y(i-1,j, IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,
				     dtdx, dtdy, FACE_XMAX, qleft);
	  
	  // Solve Riemann problem at X-interfaces and compute X-fluxes
	  riemann_hlld(qleft,qright,flux,params);

	  //
	  // store fluxes
	  //	
	  Fluxes(i,j , ID) =  flux[ID]*dtdx;
	  Fluxes(i,j , IP) =  flux[IP]*dtdx;
	  Fluxes(i,j , IU) =  flux[IU]*dtdx;
	  Fluxes(i,j , IV) =  flux[IV]*dtdx;

	} else if (dir == YDIR) {

	  // left interface : right state
	  trace_unsplit_2d_along_dir(qLoc,
				     dqX, dqY,
				     dtdx, dtdy, FACE_YMIN, qright);
	  
	  qLocNeighbor[ID] = Qdata   (i,j-1, ID);
	  dqX_neighbor[ID] = Slopes_x(i,j-1, ID);
	  dqY_neighbor[ID] = Slopes_y(i,j-1, ID);
	  
	  qLocNeighbor[IP] = Qdata   (i,j-1, IP);
	  dqX_neighbor[IP] = Slopes_x(i,j-1, IP);
	  dqY_neighbor[IP] = Slopes_y(i,j-1, IP);
	  
	  qLocNeighbor[IU] = Qdata   (i,j-1, IU);
	  dqX_neighbor[IU] = Slopes_x(i,j-1, IU);
	  dqY_neighbor[IU] = Slopes_y(i,j-1, IU);
	  
	  qLocNeighbor[IV] = Qdata   (i,j-1, IV);
	  dqX_neighbor[IV] = Slopes_x(i,j-1, IV);
	  dqY_neighbor[IV] = Slopes_y(i,j-1, IV);
	  
	  // left interface : left state
	  trace_unsplit_2d_along_dir(qLocNeighbor,
				     dqX_neighbor,dqY_neighbor,
				     dtdx, dtdy, FACE_YMAX, qleft);
	  
	  // Solve Riemann problem at Y-interfaces and compute Y-fluxes
	  swapValues(&(qleft[IU]) ,&(qleft[IV]) );
	  swapValues(&(qright[IU]),&(qright[IV]));
	  riemann_hlld(qleft,qright,flux,params);
	  
	  //
	  // update hydro array
	  //	  
	  Fluxes(i,j , ID) =  flux[ID]*dtdy;
	  Fluxes(i,j , IP) =  flux[IP]*dtdy;
	  Fluxes(i,j , IU) =  flux[IV]*dtdy; // IU/IV swapped
	  Fluxes(i,j , IV) =  flux[IU]*dtdy; // IU/IV swapped

	}
	      
    } // end if
    
  } // end operator ()
  
  DataArray2d Qdata;
  DataArray2d Slopes_x, Slopes_y;
  DataArray2d Fluxes;
  real_t dtdx, dtdy;
  
}; // ComputeTraceAndFluxes_Functor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor2D_MHD : public MHDBaseFunctor2D {

public:
  InitImplodeFunctor2D_MHD(HydroParams params,
			   DataArray2d Udata) :
    MHDBaseFunctor2D(params), Udata(Udata)  {};
  
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
    
    real_t tmp = x+y;
    if (tmp > 0.5 && tmp < 1.5) {
      Udata(i,j , ID)  = 1.0;
      Udata(i,j , IP)  = 1.0/(gamma0-1.0);
      Udata(i,j , IU)  = 0.0;
      Udata(i,j , IV)  = 0.0;
      Udata(i,j , IW)  = 0.0;
      Udata(i,j , IBX) = 0.5;
      Udata(i,j , IBY) = 0.0;
      Udata(i,j , IBZ) = 0.0;
    } else {
      Udata(i,j , ID)  = 0.125;
      Udata(i,j , IP)  = 0.14/(gamma0-1.0);
      Udata(i,j , IU)  = 0.0;
      Udata(i,j , IV)  = 0.0;
      Udata(i,j , IW)  = 0.0;
      Udata(i,j , IBX) = 0.5;
      Udata(i,j , IBY) = 0.0;
      Udata(i,j , IBZ) = 0.0;
    }
    
  } // end operator ()

  DataArray2d Udata;

}; // InitImplodeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor2D_MHD : public MHDBaseFunctor2D {

public:
  InitBlastFunctor2D_MHD(HydroParams params,
			 BlastParams bParams,
			 DataArray2d Udata) :
    MHDBaseFunctor2D(params), bParams(bParams), Udata(Udata)  {};
  
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
      Udata(i,j , ID) = blast_density_in;
      Udata(i,j , IU) = 0.0;
      Udata(i,j , IV) = 0.0;
      Udata(i,j , IW) = 0.0;
      Udata(i,j , IA) = 0.5;
      Udata(i,j , IB) = 0.5;
      Udata(i,j , IC) = 0.5;
      Udata(i,j , IP) = blast_pressure_in/(gamma0-1.0) +
	0.5* ( SQR(Udata(i,j , IA)) +
	       SQR(Udata(i,j , IB)) +
	       SQR(Udata(i,j , IC)) );
    } else {
      Udata(i,j , ID) = blast_density_out;
      Udata(i,j , IU) = 0.0;
      Udata(i,j , IV) = 0.0;
      Udata(i,j , IW) = 0.0;
      Udata(i,j , IA) = 0.5;
      Udata(i,j , IB) = 0.5;
      Udata(i,j , IC) = 0.5;
      Udata(i,j , IP) = blast_pressure_out/(gamma0-1.0) +
	0.5* ( SQR(Udata(i,j , IA)) +
	       SQR(Udata(i,j , IB)) +
	       SQR(Udata(i,j , IC)) );
    }
    
  } // end operator ()
  
  DataArray2d Udata;
  BlastParams bParams;
  
}; // InitBlastFunctor2D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
enum OrszagTang_init_type {
  INIT_ALL_VAR_BUT_ENERGY = 0,
  INIT_ENERGY = 1
};
template<OrszagTang_init_type ot_type>
class InitOrszagTangFunctor2D : public MHDBaseFunctor2D {

public:
  InitOrszagTangFunctor2D(HydroParams params,
			  DataArray2d Udata) :
    MHDBaseFunctor2D(params), Udata(Udata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

    if (ot_type == INIT_ALL_VAR_BUT_ENERGY)
      init_all_var_but_energy(index);
    else if(ot_type == INIT_ENERGY)
      init_energy(index);

  } // end operator ()

  KOKKOS_INLINE_FUNCTION
  void init_all_var_but_energy(const int index) const
  {
    
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    const double xmin = params.xmin;
    const double ymin = params.ymin;
        
    const double dx = params.dx;
    const double dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
    const double TwoPi = 4.0*asin(1.0);
    const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = gamma0/(2.0*TwoPi);
    const double d0    = gamma0*p0;
    const double v0    = 1.0;

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    double xPos = xmin + dx/2 + (i-ghostWidth)*dx;
    double yPos = ymin + dy/2 + (j-ghostWidth)*dy;
    
    if(j < jsize  &&
       i < isize ) {

      // density
      Udata(i,j,ID) = d0;
      
      // rho*vx
      Udata(i,j,IU)  = static_cast<real_t>(-d0*v0*sin(yPos*TwoPi));
      
      // rho*vy
      Udata(i,j,IV)  = static_cast<real_t>( d0*v0*sin(xPos*TwoPi));
      
      // rho*vz
      Udata(i,j,IW) =  ZERO_F;
      
      // bx, by, bz
      Udata(i,j, IBX) = -B0*sin(    yPos*TwoPi);
      Udata(i,j, IBY) =  B0*sin(2.0*xPos*TwoPi);
      Udata(i,j, IBZ) =  0.0;

    }
    
  } // init_all_var_but_energy

  KOKKOS_INLINE_FUNCTION
  void init_energy(const int index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    //const double xmin = params.xmin;
    //const double ymin = params.ymin;
    
    //const double dx = params.dx;
    //const double dy = params.dy;
    
    const real_t gamma0 = params.settings.gamma0;
    
    const double TwoPi = 4.0*asin(1.0);
    const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = gamma0/(2.0*TwoPi);
    //const double d0    = gamma0*p0;
    //const double v0    = 1.0;

    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    //double xPos = xmin + dx/2 + (i-ghostWidth)*dx;
    //double yPos = ymin + dy/2 + (j-ghostWidth)*dy;
        
    if (i<isize-1 and j<jsize-1) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(i+1,j,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i,j+1,IBY)) );
    } else if ( (i <isize-1) and (j==jsize-1)) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(i+1,j           ,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i  ,2*ghostWidth,IBY)) );
    } else if ( (i==isize-1) and (j <jsize-1)) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(2*ghostWidth,j  ,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i           ,j+1,IBY)) );
    } else if ( (i==isize-1) and (j==jsize-1) ) {
      Udata(i,j,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,IU)) / Udata(i,j,ID) +
		SQR(Udata(i,j,IV)) / Udata(i,j,ID) +
		0.25*SQR(Udata(i,j,IBX) + Udata(2*ghostWidth,j ,IBX)) + 
		0.25*SQR(Udata(i,j,IBY) + Udata(i,2*ghostWidth ,IBY)) );
    }
    
  } // init_energy
  
  DataArray2d Udata;
  
}; // InitOrszagTangFunctor2D

} // namespace muscl
} // namespace ppkMHD

#endif // MHD_RUN_FUNCTORS_2D_H_

