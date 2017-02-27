#ifndef MHD_RUN_FUNCTORS_3D_H_
#define MHD_RUN_FUNCTORS_3D_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "MHDBaseFunctor3D.h"

#include "BlastParams.h"

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

namespace ppkMHD { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor3D_MHD : public MHDBaseFunctor3D {

public:
  
  ComputeDtFunctor3D_MHD(HydroParams params,
			 DataArray3d Qdata) :
    MHDBaseFunctor3D(params),
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
      
      MHDState qLoc; // primitive    variables in current cell
      
      // get primitive variables in current cell
      qLoc.d  = Qdata(i,j,k,ID);
      qLoc.p  = Qdata(i,j,k,IP);
      qLoc.u  = Qdata(i,j,k,IU);
      qLoc.v  = Qdata(i,j,k,IV);
      qLoc.w  = Qdata(i,j,k,IW);
      qLoc.bx = Qdata(i,j,k,IBX);
      qLoc.by = Qdata(i,j,k,IBY);
      qLoc.bz = Qdata(i,j,k,IBZ);

      // compute fastest information speeds
      real_t fastInfoSpeed[3];
      find_speed_info<THREE_D>(qLoc, fastInfoSpeed);
      
      real_t vx = fastInfoSpeed[IX];
      real_t vy = fastInfoSpeed[IY];
      real_t vz = fastInfoSpeed[IZ];
      
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

  DataArray3d Qdata;
  
}; // ComputeDtFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  ConvertToPrimitivesFunctor3D_MHD(HydroParams params,
				   DataArray3d Udata,
				   DataArray3d Qdata) :
    MHDBaseFunctor3D(params), Udata(Udata), Qdata(Qdata)  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // magnetic field in neighbor cells
    real_t magFieldNeighbors[3];
    
    if(k >= 0 && k < ksize-1  &&
       j >= 0 && j < jsize-1  &&
       i >= 0 && i < isize-1 ) {
      
      MHDState uLoc; // conservative    variables in current cell
      MHDState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc.d  = Udata(i,j,k,ID);
      uLoc.p  = Udata(i,j,k,IP);
      uLoc.u  = Udata(i,j,k,IU);
      uLoc.v  = Udata(i,j,k,IV);
      uLoc.w  = Udata(i,j,k,IW);
      uLoc.bx = Udata(i,j,k,IBX);
      uLoc.by = Udata(i,j,k,IBY);
      uLoc.bz = Udata(i,j,k,IBZ);

      // get mag field in neighbor cells
      magFieldNeighbors[IX] = Udata(i+1,j  ,k  ,IBX);
      magFieldNeighbors[IY] = Udata(i  ,j+1,k  ,IBY);
      magFieldNeighbors[IZ] = Udata(i  ,j  ,k+1,IBZ);
      
      // get primitive variables in current cell
      constoprim_mhd(uLoc, magFieldNeighbors, c, qLoc);

      // copy q state in q global
      Qdata(i,j,k,ID)  = qLoc.d;
      Qdata(i,j,k,IP)  = qLoc.p;
      Qdata(i,j,k,IU)  = qLoc.u;
      Qdata(i,j,k,IV)  = qLoc.v;
      Qdata(i,j,k,IW)  = qLoc.w;
      Qdata(i,j,k,IBX) = qLoc.bx;
      Qdata(i,j,k,IBY) = qLoc.by;
      Qdata(i,j,k,IBZ) = qLoc.bz;
      
    }
    
  }
  
  DataArray3d Udata;
  DataArray3d Qdata;
    
}; // ConvertToPrimitivesFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeElecFieldFunctor3D : public MHDBaseFunctor3D {

public:

  ComputeElecFieldFunctor3D(HydroParams params,
			    DataArray3d Udata,
			    DataArray3d Qdata,
			    DataArrayVector3 ElecField) :
    MHDBaseFunctor3D(params),
    Udata(Udata), Qdata(Qdata), ElecField(ElecField) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    if (k > 0 && k < ksize-1 &&
	j > 0 && j < jsize-1 &&
	i > 0 && i < isize-1) {
      
      real_t u, v, w, A, B, C;
      
      // compute Ex
      v = ONE_FOURTH_F * ( Qdata(i  ,j-1,k-1,IV) +
			   Qdata(i  ,j-1,k  ,IV) +
			   Qdata(i  ,j  ,k-1,IV) +
			   Qdata(i  ,j  ,k  ,IV) );
      
      w = ONE_FOURTH_F * ( Qdata(i  ,j-1,k-1,IW) +
			   Qdata(i  ,j-1,k  ,IW) +
			   Qdata(i  ,j  ,k-1,IW) +
			   Qdata(i  ,j  ,k  ,IW) );
      
      B = HALF_F  * ( Udata(i  ,j  ,k-1,IB) +
		      Udata(i  ,j  ,k  ,IB) );
      
      C = HALF_F  * ( Udata(i  ,j-1,k  ,IC) +
		      Udata(i  ,j  ,k  ,IC) );
      
      ElecField(i,j,k,IX) = v*C-w*B;
      
      // compute Ey
      u = ONE_FOURTH_F * ( Qdata   (i-1,j  ,k-1,IU) +
			   Qdata   (i-1,j  ,k  ,IU) +
			   Qdata   (i  ,j  ,k-1,IU) +
			   Qdata   (i  ,j  ,k  ,IU) );
      
      w = ONE_FOURTH_F * ( Qdata   (i-1,j  ,k-1,IW) +
			   Qdata   (i-1,j  ,k  ,IW) +
			   Qdata   (i  ,j  ,k-1,IW) +
			   Qdata   (i  ,j  ,k  ,IW) );
      
      A = HALF_F  * ( Udata(i  ,j  ,k-1,IA) +
		      Udata(i  ,j  ,k  ,IA) );
      
      C = HALF_F  * ( Udata(i-1,j  ,k  ,IC) +
		      Udata(i  ,j  ,k  ,IC) );
      
      ElecField(i,j,k,IY) = w*A-u*C;
      
      // compute Ez
      u = ONE_FOURTH_F * ( Qdata   (i-1,j-1,k  ,IU) +
			   Qdata   (i-1,j  ,k  ,IU) +
			   Qdata   (i  ,j-1,k  ,IU) +
			   Qdata   (i  ,j  ,k  ,IU) );
      
      v = ONE_FOURTH_F * ( Qdata   (i-1,j-1,k  ,IV) +
			   Qdata   (i-1,j  ,k  ,IV) +
			   Qdata   (i  ,j-1,k  ,IV) +
			   Qdata   (i  ,j  ,k  ,IV) );
      
      A = HALF_F  * ( Udata(i  ,j-1,k  ,IA) +
		      Udata(i  ,j  ,k  ,IA) );
      
      B = HALF_F  * ( Udata(i-1,j  ,k  ,IB) +
		      Udata(i  ,j  ,k  ,IB) );
      
      ElecField(i,j,k,IZ) = u*B-v*A;
      
    }
  } // operator ()

  DataArray3d Udata;
  DataArray3d Qdata;  
  DataArrayVector3 ElecField;
  
}; // ComputeElecFieldFunctor3D
  
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeMagSlopesFunctor3D : public MHDBaseFunctor3D {

public:

  ComputeMagSlopesFunctor3D(HydroParams      params,
			    DataArray3d      Udata,
			    DataArrayVector3 DeltaA,
			    DataArrayVector3 DeltaB,
			    DataArrayVector3 DeltaC) :
    MHDBaseFunctor3D(params), Udata(Udata),
    DeltaA(DeltaA), DeltaB(DeltaB), DeltaC(DeltaC) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    //const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    if (k > 0 && k < ksize-1 &&
	j > 0 && j < jsize-1 &&
	i > 0 && i < isize-1) {

      real_t bfSlopes[15];
      real_t dbfSlopes[3][3];
      
      real_t (&dbfX)[3] = dbfSlopes[IX];
      real_t (&dbfY)[3] = dbfSlopes[IY];
      real_t (&dbfZ)[3] = dbfSlopes[IZ];
      
      // get magnetic slopes dbf
      bfSlopes[0]  = Udata(i  ,j  ,k  , IA);
      bfSlopes[1]  = Udata(i  ,j+1,k  , IA);
      bfSlopes[2]  = Udata(i  ,j-1,k  , IA);
      bfSlopes[3]  = Udata(i  ,j  ,k+1, IA);
      bfSlopes[4]  = Udata(i  ,j  ,k-1, IA);
      
      bfSlopes[5]  = Udata(i  ,j  ,k  , IB);
      bfSlopes[6]  = Udata(i+1,j  ,k  , IB);
      bfSlopes[7]  = Udata(i-1,j  ,k  , IB);
      bfSlopes[8]  = Udata(i  ,j  ,k+1, IB);
      bfSlopes[9]  = Udata(i  ,j  ,k-1, IB);
      
      bfSlopes[10] = Udata(i  ,j  ,k  , IC);
      bfSlopes[11] = Udata(i+1,j  ,k  , IC);
      bfSlopes[12] = Udata(i-1,j  ,k  , IC);
      bfSlopes[13] = Udata(i  ,j+1,k  , IC);
      bfSlopes[14] = Udata(i  ,j-1,k  , IC);
      
      // compute magnetic slopes
      slope_unsplit_mhd_3d(bfSlopes, dbfSlopes);
      
      // store magnetic slopes
      DeltaA(i,j,k,0) = dbfX[IX];
      DeltaA(i,j,k,1) = dbfY[IX];
      DeltaA(i,j,k,2) = dbfZ[IX];
      
      DeltaB(i,j,k,0) = dbfX[IY];
      DeltaB(i,j,k,1) = dbfY[IY];
      DeltaB(i,j,k,2) = dbfZ[IY];
      
      DeltaC(i,j,k,0) = dbfX[IZ];
      DeltaC(i,j,k,1) = dbfY[IZ];
      DeltaC(i,j,k,2) = dbfZ[IZ];
      
    }

  } // operator ()
  DataArray3d Udata;
  DataArrayVector3 DeltaA;
  DataArrayVector3 DeltaB;
  DataArrayVector3 DeltaC;
    
}; // class ComputeMagSlopesFunctor3D

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  ComputeTraceFunctor3D_MHD(HydroParams params,
			    DataArray3d Udata,
			    DataArray3d Qdata,
			    DataArrayVector3 DeltaA,
			    DataArrayVector3 DeltaB,
			    DataArrayVector3 DeltaC,
			    DataArrayVector3 ElecField,
			    DataArray3d Qm_x,
			    DataArray3d Qm_y,
			    DataArray3d Qm_z,
			    DataArray3d Qp_x,
			    DataArray3d Qp_y,
			    DataArray3d Qp_z,
			    DataArray3d QEdge_RT,
			    DataArray3d QEdge_RB,
			    DataArray3d QEdge_LT,
			    DataArray3d QEdge_LB,
			    DataArray3d QEdge_RT2,
			    DataArray3d QEdge_RB2,
			    DataArray3d QEdge_LT2,
			    DataArray3d QEdge_LB2,
			    DataArray3d QEdge_RT3,
			    DataArray3d QEdge_RB3,
			    DataArray3d QEdge_LT3,
			    DataArray3d QEdge_LB3,
			    real_t dtdx,
			    real_t dtdy,
			    real_t dtdz) :
    MHDBaseFunctor3D(params),
    Udata(Udata), Qdata(Qdata),
    DeltaA(DeltaA), DeltaB(DeltaB), DeltaC(DeltaC), ElecField(ElecField),
    Qm_x(Qm_x), Qm_y(Qm_y), Qm_z(Qm_z),
    Qp_x(Qp_x), Qp_y(Qp_y), Qp_z(Qp_z),
    QEdge_RT (QEdge_RT),  QEdge_RB (QEdge_RB),  QEdge_LT (QEdge_LT),  QEdge_LB (QEdge_LB),
    QEdge_RT2(QEdge_RT2), QEdge_RB2(QEdge_RB2), QEdge_LT2(QEdge_LT2), QEdge_LB2(QEdge_LB2),
    QEdge_RT3(QEdge_RT3), QEdge_RB3(QEdge_RB3), QEdge_LT3(QEdge_LT3), QEdge_LB3(QEdge_LB3),
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
    
    if(k >= ghostWidth-2 && k < ksize-ghostWidth+1 &&
       j >= ghostWidth-2 && j < jsize-ghostWidth+1 &&
       i >= ghostWidth-2 && i < isize-ghostWidth+1) {

      MHDState q;
      MHDState qPlusX, qMinusX, qPlusY, qMinusY, qPlusZ, qMinusZ;
      MHDState dq[3];
      
      real_t bfNb[6];
      real_t dbf[12];
      
      real_t elecFields[3][2][2];
      // alias to electric field components
      real_t (&Ex)[2][2] = elecFields[IX];
      real_t (&Ey)[2][2] = elecFields[IY];
      real_t (&Ez)[2][2] = elecFields[IZ];
      
      MHDState qm[THREE_D];
      MHDState qp[THREE_D];
      MHDState qEdge[4][3]; // array for qRT, qRB, qLT, qLB
      
      real_t xPos = params.xmin + params.dx/2 + (i-ghostWidth)*params.dx;
      
      // get primitive variables state vector
      get_state(Qdata, i  ,j  ,k  , q      );
      get_state(Qdata, i+1,j  ,k  , qPlusX );
      get_state(Qdata, i-1,j  ,k  , qMinusX);
      get_state(Qdata, i  ,j+1,k  , qPlusY );
      get_state(Qdata, i  ,j-1,k  , qMinusY);
      get_state(Qdata, i  ,j  ,k+1, qPlusZ );
      get_state(Qdata, i  ,j  ,k-1, qMinusZ);
      
      // get hydro slopes dq
      slope_unsplit_hydro_3d(q, 
			     qPlusX, qMinusX, 
			     qPlusY, qMinusY, 
			     qPlusZ, qMinusZ,
			     dq);
      
      // get face-centered magnetic components
      bfNb[0] = Udata(i  ,j  ,k  , IA);
      bfNb[1] = Udata(i+1,j  ,k  , IA);
      bfNb[2] = Udata(i  ,j  ,k  , IB);
      bfNb[3] = Udata(i  ,j+1,k  , IB);
      bfNb[4] = Udata(i  ,j  ,k  , IC);
      bfNb[5] = Udata(i  ,j  ,k+1, IC);
      
      // get dbf (transverse magnetic slopes)
      dbf[0]  = DeltaA(i  ,j  ,k  , IY);
      dbf[1]  = DeltaA(i  ,j  ,k  , IZ);
      dbf[2]  = DeltaB(i  ,j  ,k  , IX);
      dbf[3]  = DeltaB(i  ,j  ,k  , IZ);
      dbf[4]  = DeltaC(i  ,j  ,k  , IX);
      dbf[5]  = DeltaC(i  ,j  ,k  , IY);
      
      dbf[6]  = DeltaA(i+1,j  ,k  , IY);
      dbf[7]  = DeltaA(i+1,j  ,k  , IZ);
      dbf[8]  = DeltaB(i  ,j+1,k  , IX);
      dbf[9]  = DeltaB(i  ,j+1,k  , IZ);
      dbf[10] = DeltaC(i  ,j  ,k+1, IX);
      dbf[11] = DeltaC(i  ,j  ,k+1, IY);
      
      // get electric field components
      Ex[0][0] = ElecField(i  ,j  ,k  , IX);
      Ex[0][1] = ElecField(i  ,j  ,k+1, IX);
      Ex[1][0] = ElecField(i  ,j+1,k  , IX);
      Ex[1][1] = ElecField(i  ,j+1,k+1, IX);
      
      Ey[0][0] = ElecField(i  ,j  ,k  , IY);
      Ey[0][1] = ElecField(i  ,j  ,k+1, IY);
      Ey[1][0] = ElecField(i+1,j  ,k  , IY);
      Ey[1][1] = ElecField(i+1,j  ,k+1, IY);
      
      Ez[0][0] = ElecField(i  ,j  ,k  , IZ);
      Ez[0][1] = ElecField(i  ,j+1,k  , IZ);
      Ez[1][0] = ElecField(i+1,j  ,k  , IZ);
      Ez[1][1] = ElecField(i+1,j+1,k  , IZ);
      
      // compute qm, qp and qEdge
      trace_unsplit_mhd_3d_simpler(q, dq, bfNb, dbf, elecFields, 
				   dtdx, dtdy, dtdz, xPos,
				   qm, qp, qEdge);
      
      // gravity predictor / modify velocity components
      // if (gravityEnabled) { 
	
      // 	real_t grav_x = HALF_F * dt * h_gravity(i,j,k,IX);
      // 	real_t grav_y = HALF_F * dt * h_gravity(i,j,k,IY);
      // 	real_t grav_z = HALF_F * dt * h_gravity(i,j,k,IZ);
	
      // 	qm[0][IU] += grav_x; qm[0][IV] += grav_y; qm[0][IW] += grav_z;
      // 	qp[0][IU] += grav_x; qp[0][IV] += grav_y; qp[0][IW] += grav_z;
	
      // 	qm[1][IU] += grav_x; qm[1][IV] += grav_y; qm[1][IW] += grav_z;
      // 	qp[1][IU] += grav_x; qp[1][IV] += grav_y; qp[1][IW] += grav_z;
	
      // 	qm[2][IU] += grav_x; qm[2][IV] += grav_y; qm[2][IW] += grav_z;
      // 	qp[2][IU] += grav_x; qp[2][IV] += grav_y; qp[2][IW] += grav_z;
	
      // 	qEdge[IRT][0][IU] += grav_x;
      // 	qEdge[IRT][0][IV] += grav_y;
      // 	qEdge[IRT][0][IW] += grav_z;
      // 	qEdge[IRT][1][IU] += grav_x;
      // 	qEdge[IRT][1][IV] += grav_y;
      // 	qEdge[IRT][1][IW] += grav_z;
      // 	qEdge[IRT][2][IU] += grav_x;
      // 	qEdge[IRT][2][IV] += grav_y;
      // 	qEdge[IRT][2][IW] += grav_z;
	
      // 	qEdge[IRB][0][IU] += grav_x;
      // 	qEdge[IRB][0][IV] += grav_y;
      // 	qEdge[IRB][0][IW] += grav_z;
      // 	qEdge[IRB][1][IU] += grav_x;
      // 	qEdge[IRB][1][IV] += grav_y;
      // 	qEdge[IRB][1][IW] += grav_z;
      // 	qEdge[IRB][2][IU] += grav_x;
      // 	qEdge[IRB][2][IV] += grav_y;
      // 	qEdge[IRB][2][IW] += grav_z;
	
      // 	qEdge[ILT][0][IU] += grav_x;
      // 	qEdge[ILT][0][IV] += grav_y;
      // 	qEdge[ILT][0][IW] += grav_z;
      // 	qEdge[ILT][1][IU] += grav_x;
      // 	qEdge[ILT][1][IV] += grav_y;
      // 	qEdge[ILT][1][IW] += grav_z;
      // 	qEdge[ILT][2][IU] += grav_x;
      // 	qEdge[ILT][2][IV] += grav_y;
      // 	qEdge[ILT][2][IW] += grav_z;
	
      // 	qEdge[ILB][0][IU] += grav_x;
      // 	qEdge[ILB][0][IV] += grav_y;
      // 	qEdge[ILB][0][IW] += grav_z;
      // 	qEdge[ILB][1][IU] += grav_x;
      // 	qEdge[ILB][1][IV] += grav_y;
      // 	qEdge[ILB][1][IW] += grav_z;
      // 	qEdge[ILB][2][IU] += grav_x;
      // 	qEdge[ILB][2][IV] += grav_y;
      // 	qEdge[ILB][2][IW] += grav_z;
	
      // } // end gravity predictor
      
      // store qm, qp, qEdge : only what is really needed
      set_state(Qm_x, i,j,k, qm[0]);
      set_state(Qp_x, i,j,k, qp[0]);
      set_state(Qm_y, i,j,k, qm[1]);
      set_state(Qp_y, i,j,k, qp[1]);
      set_state(Qm_z, i,j,k, qm[2]);
      set_state(Qp_z, i,j,k, qp[2]);
      
      set_state(QEdge_RT , i,j,k, qEdge[IRT][0]); 
      set_state(QEdge_RB , i,j,k, qEdge[IRB][0]); 
      set_state(QEdge_LT , i,j,k, qEdge[ILT][0]); 
      set_state(QEdge_LB , i,j,k, qEdge[ILB][0]); 
      
      set_state(QEdge_RT2, i,j,k, qEdge[IRT][1]); 
      set_state(QEdge_RB2, i,j,k, qEdge[IRB][1]); 
      set_state(QEdge_LT2, i,j,k, qEdge[ILT][1]); 
      set_state(QEdge_LB2, i,j,k, qEdge[ILB][1]); 
      
      set_state(QEdge_RT3, i,j,k, qEdge[IRT][2]); 
      set_state(QEdge_RB3, i,j,k, qEdge[IRB][2]); 
      set_state(QEdge_LT3, i,j,k, qEdge[ILT][2]); 
      set_state(QEdge_LB3, i,j,k, qEdge[ILB][2]); 
      
    }
    
  } // operator ()
      
  DataArray3d Udata, Qdata;
  DataArrayVector3 DeltaA, DeltaB, DeltaC, ElecField;
  DataArray3d Qm_x, Qm_y, Qm_z;
  DataArray3d Qp_x, Qp_y, Qp_z;
  DataArray3d QEdge_RT,  QEdge_RB,  QEdge_LT,  QEdge_LB;
  DataArray3d QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray3d QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  real_t dtdx, dtdy, dtdz;
  
}; // class ComputeTraceFunctor3D_MHD
  
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndStoreFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  ComputeFluxesAndStoreFunctor3D_MHD(HydroParams params,
				     DataArray3d Qm_x,
				     DataArray3d Qm_y,
				     DataArray3d Qm_z,
				     DataArray3d Qp_x,
				     DataArray3d Qp_y,
				     DataArray3d Qp_z,
				     DataArray3d Fluxes_x,
				     DataArray3d Fluxes_y,
				     DataArray3d Fluxes_z,
				     real_t dtdx,
				     real_t dtdy,
				     real_t dtdz) :
    MHDBaseFunctor3D(params),
    Qm_x(Qm_x), Qm_y(Qm_y), Qm_z(Qm_z),
    Qp_x(Qp_x), Qp_y(Qp_y), Qp_z(Qp_z),
    Fluxes_x(Fluxes_x), Fluxes_y(Fluxes_y), Fluxes_z(Fluxes_z),
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
    
    if(k >= ghostWidth && k < ksize - ghostWidth+1 &&
       j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {
      
      MHDState qleft, qright;
      MHDState flux;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //
      get_state(Qm_x, i-1,j  ,k, qleft);      
      get_state(Qp_x, i  ,j  ,k, qright);
      
      // compute hydro flux along X
      riemann_hlld(qleft,qright,flux);

      // store fluxes
      set_state(Fluxes_x, i, j, k, flux);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      get_state(Qm_y, i,j-1,k, qleft);
      swapValues(&(qleft.u)  ,&(qleft.v) );
      swapValues(&(qleft.bx) ,&(qleft.by) );

      get_state(Qp_y, i,j,k, qright);
      swapValues(&(qright.u)  ,&(qright.v) );
      swapValues(&(qright.bx) ,&(qright.by) );
      
      // compute hydro flux along Y
      riemann_hlld(qleft,qright,flux);
            
      // store fluxes
      set_state(Fluxes_y, i,j,k, flux);
      
      //
      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      //
      get_state(Qm_z, i,j,k-1, qleft);
      swapValues(&(qleft.u)  ,&(qleft.w) );
      swapValues(&(qleft.bx) ,&(qleft.bz) );

      get_state(Qp_z, i,j,k, qright);
      swapValues(&(qright.u)  ,&(qright.w) );
      swapValues(&(qright.bx) ,&(qright.bz) );
      
      // compute hydro flux along Z
      riemann_hlld(qleft,qright,flux);
            
      // store fluxes
      set_state(Fluxes_z, i,j,k, flux);

    }
    
  }
  
  DataArray3d Qm_x, Qm_y, Qm_z;
  DataArray3d Qp_x, Qp_y, Qp_z;
  DataArray3d Fluxes_x, Fluxes_y, Fluxes_z;
  real_t dtdx, dtdy, dtdz;
  
}; // ComputeFluxesAndStoreFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndStoreFunctor3D : public MHDBaseFunctor3D {
  
public:
  
  ComputeEmfAndStoreFunctor3D(HydroParams params,
			      DataArray3d QEdge_RT,
			      DataArray3d QEdge_RB,
			      DataArray3d QEdge_LT,
			      DataArray3d QEdge_LB,
			      DataArray3d QEdge_RT2,
			      DataArray3d QEdge_RB2,
			      DataArray3d QEdge_LT2,
			      DataArray3d QEdge_LB2,
			      DataArray3d QEdge_RT3,
			      DataArray3d QEdge_RB3,
			      DataArray3d QEdge_LT3,
			      DataArray3d QEdge_LB3,
			      DataArrayVector3 Emf,
			      real_t dtdx,
			      real_t dtdy,
			      real_t dtdz) :
    MHDBaseFunctor3D(params),
    QEdge_RT(QEdge_RT),   QEdge_RB(QEdge_RB),   QEdge_LT(QEdge_LT),   QEdge_LB(QEdge_LB),
    QEdge_RT2(QEdge_RT2), QEdge_RB2(QEdge_RB2), QEdge_LT2(QEdge_LT2), QEdge_LB2(QEdge_LB2),
    QEdge_RT3(QEdge_RT3), QEdge_RB3(QEdge_RB3), QEdge_LT3(QEdge_LT3), QEdge_LB3(QEdge_LB3),
    Emf(Emf),
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
    
    if(k >= ghostWidth && k < ksize - ghostWidth+1 &&
       j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {

      MHDState qEdge_emf[4];

      // preparation for calling compute_emf (equivalent to cmp_mag_flx
      // in DUMSES)
      // in the following, the 2 first indexes in qEdge_emf array play
      // the same offset role as in the calling argument of cmp_mag_flx 
      // in DUMSES (if you see what I mean ?!)

      // actually compute emfZ 
      get_state(QEdge_RT3, i-1,j-1,k  , qEdge_emf[IRT]);
      get_state(QEdge_RB3, i-1,j  ,k  , qEdge_emf[IRB]); 
      get_state(QEdge_LT3, i  ,j-1,k  , qEdge_emf[ILT]);
      get_state(QEdge_LB3, i  ,j  ,k  , qEdge_emf[ILB]);

      Emf(i,j,k,I_EMFZ) = compute_emf<EMFZ>(qEdge_emf);
      
      // actually compute emfY (take care that RB and LT are
      // swapped !!!)
      get_state(QEdge_RT2, i-1,j  ,k-1, qEdge_emf[IRT]);
      get_state(QEdge_LT2, i  ,j  ,k-1, qEdge_emf[IRB]); 
      get_state(QEdge_RB2, i-1,j  ,k  , qEdge_emf[ILT]);
      get_state(QEdge_LB2, i  ,j  ,k  , qEdge_emf[ILB]);

      Emf(i,j,k,I_EMFY) = compute_emf<EMFY>(qEdge_emf);
      
      // actually compute emfX
      get_state(QEdge_RT, i  ,j-1,k-1, qEdge_emf[IRT]);
      get_state(QEdge_RB, i  ,j-1,k  , qEdge_emf[IRB]); 
      get_state(QEdge_LT, i  ,j  ,k-1, qEdge_emf[ILT]);
      get_state(QEdge_LB, i  ,j  ,k  , qEdge_emf[ILB]);

      Emf(i,j,k,I_EMFX) = compute_emf<EMFX>(qEdge_emf);
    }
  }

  DataArray3d QEdge_RT,  QEdge_RB,  QEdge_LT,  QEdge_LB;
  DataArray3d QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray3d QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  DataArrayVector3 Emf;
  real_t dtdx, dtdy, dtdz;

}; // ComputeEmfAndStoreFunctor3D

  
/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor3D_MHD : public MHDBaseFunctor3D {

public:

  UpdateFunctor3D_MHD(HydroParams params,
		      DataArray3d Udata,
		      DataArray3d FluxData_x,
		      DataArray3d FluxData_y,
		      DataArray3d FluxData_z,
		      real_t dtdx,
		      real_t dtdy,
		      real_t dtdz) :
    MHDBaseFunctor3D(params),
    Udata(Udata), 
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

    if(k >= ghostWidth && k < ksize-ghostWidth  &&
       j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      MHDState udata;
      MHDState flux;
      get_state(Udata, i,j,k, udata);

      // add up contributions from all 6 faces
      
      get_state(FluxData_x, i  ,j  ,k  , flux);      
      udata.d  +=  flux.d*dtdx;
      udata.p  +=  flux.p*dtdx;
      udata.u  +=  flux.u*dtdx;
      udata.v  +=  flux.v*dtdx;
      udata.w  +=  flux.w*dtdx;
      
      get_state(FluxData_x, i+1,j  ,k  , flux);
      udata.d  -=  flux.d*dtdx;
      udata.p  -=  flux.p*dtdx;
      udata.u  -=  flux.u*dtdx;
      udata.v  -=  flux.v*dtdx;
      udata.w  -=  flux.w*dtdx;
      
      get_state(FluxData_y, i  ,j  ,k  , flux);
      udata.d  +=  flux.d*dtdy;
      udata.p  +=  flux.p*dtdy;
      udata.u  +=  flux.v*dtdy; //
      udata.v  +=  flux.u*dtdy; //
      udata.w  +=  flux.w*dtdy;
      
      get_state(FluxData_y, i  ,j+1,k  , flux);
      udata.d  -=  flux.d*dtdy;
      udata.p  -=  flux.p*dtdy;
      udata.u  -=  flux.v*dtdy; //
      udata.v  -=  flux.u*dtdy; //
      udata.w  -=  flux.w*dtdy;

      get_state(FluxData_z, i  ,j  ,k  , flux);
      udata.d  +=  flux.d*dtdy;
      udata.p  +=  flux.p*dtdy;
      udata.u  +=  flux.w*dtdy; //
      udata.v  +=  flux.v*dtdy;
      udata.w  +=  flux.u*dtdy; //

      get_state(FluxData_z, i  ,j  ,k+1, flux);
      udata.d  -=  flux.d*dtdz;
      udata.p  -=  flux.p*dtdz;
      udata.u  -=  flux.w*dtdz; //
      udata.v  -=  flux.v*dtdz;
      udata.w  -=  flux.u*dtdz; //
      
      // write back result in Udata
      set_state(Udata, i  ,j  ,k  , udata);
      
    } // end if
    
  } // end operator ()
  
  DataArray3d Udata;
  DataArray3d FluxData_x, FluxData_y, FluxData_z;
  real_t dtdx, dtdy, dtdz;
  
}; // UpdateFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateEmfFunctor3D : public MHDBaseFunctor3D {

public:

  UpdateEmfFunctor3D(HydroParams params,
		     DataArray3d Udata,
		     DataArrayVector3 Emf,
		     real_t dtdx,
		     real_t dtdy,
		     real_t dtdz) :
    MHDBaseFunctor3D(params),
    Udata(Udata), 
    Emf(Emf),
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

    if(k >= ghostWidth && k < ksize-ghostWidth+1  &&
       j >= ghostWidth && j < jsize-ghostWidth+1  &&
       i >= ghostWidth && i < isize-ghostWidth+1 ) {

      MHDState udata;
      get_state(Udata, i,j,k, udata);

      if (k<ksize-ghostWidth) {
	udata.bx += ( Emf(i  ,j+1, k,  I_EMFZ) - 
		      Emf(i,  j  , k,  I_EMFZ) ) * dtdy;
	
	udata.by -= ( Emf(i+1,j  , k,  I_EMFZ) - 
		      Emf(i  ,j  , k,  I_EMFZ) ) * dtdx;
	
      }
      
      // update BX
      udata.bx -= ( Emf(i  ,j  ,k+1,  I_EMFY) -
		    Emf(i  ,j  ,k  ,  I_EMFY) ) * dtdz;
      
      // update BY
      udata.by += ( Emf(i  ,j  ,k+1,  I_EMFX) -
		    Emf(i  ,j  ,k  ,  I_EMFX) ) * dtdz;
      
      // update BZ
      udata.bz += ( Emf(i+1,j  ,k  ,  I_EMFY) -
		    Emf(i  ,j  ,k  ,  I_EMFY) ) * dtdx;
      
      udata.bz -= ( Emf(i  ,j+1,k  ,  I_EMFX) -
		    Emf(i  ,j  ,k  ,  I_EMFX) ) * dtdy;

      Udata(i,j,k, IA) = udata.bx;
      Udata(i,j,k, IB) = udata.by;
      Udata(i,j,k, IC) = udata.bz;
      
    }
  } // operator()

  DataArray3d Udata;
  DataArrayVector3 Emf;
  real_t dtdx, dtdy, dtdz;

}; // UpdateEmfFunctor3D


/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor : public MHDBaseFunctor3D {

public:
  InitImplodeFunctor(HydroParams params,
		     DataArray3d Udata) :
    MHDBaseFunctor3D(params), Udata(Udata)  {};
  
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
    
    real_t tmp = x+y+z;
    if (tmp > 0.5 && tmp < 2.5) {
      Udata(i,j,k , ID)  = 1.0;
      Udata(i,j,k , IU)  = 0.0;
      Udata(i,j,k , IV)  = 0.0;
      Udata(i,j,k , IW)  = 0.0;
      Udata(i,j,k , IBX) = 0.5;
      Udata(i,j,k , IBY) = 0.0;
      Udata(i,j,k , IBZ) = 0.0;
      Udata(i,j,k , IP)  = 1.0/(gamma0-1.0) +
	0.5* ( SQR(Udata(i,j,k , IBX)) +
	       SQR(Udata(i,j,k , IBY)) +
	       SQR(Udata(i,j,k , IBZ)) );
    } else {
      Udata(i,j,k , ID)  = 0.125;
      Udata(i,j,k , IU)  = 0.0;
      Udata(i,j,k , IV)  = 0.0;
      Udata(i,j,k , IW)  = 0.0;
      Udata(i,j,k , IBX) = 0.5;
      Udata(i,j,k , IBY) = 0.0;
      Udata(i,j,k , IBZ) = 0.0;
      Udata(i,j,k , IP)  = 0.14/(gamma0-1.0)  +
	0.5* ( SQR(Udata(i,j,k , IBX)) +
	       SQR(Udata(i,j,k , IBY)) +
	       SQR(Udata(i,j,k , IBZ)) );
    }
    
  } // end operator ()

  DataArray3d Udata;

}; // InitImplodeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor3D_MHD : public MHDBaseFunctor3D {

public:
  InitBlastFunctor3D_MHD(HydroParams params,
			 BlastParams bParams,
			 DataArray3d Udata) :
    MHDBaseFunctor3D(params), bParams(bParams), Udata(Udata)  {};
  
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
      Udata(i,j,k , ID) = blast_density_in;
      Udata(i,j,k , IU) = 0.0;
      Udata(i,j,k , IV) = 0.0;
      Udata(i,j,k , IW) = 0.0;
      Udata(i,j,k , IA) = 0.5;
      Udata(i,j,k , IB) = 0.5;
      Udata(i,j,k , IC) = 0.5;
      Udata(i,j,k , IP) = blast_pressure_in/(gamma0-1.0) +
	0.5* ( SQR(Udata(i,j,k , IA)) +
	       SQR(Udata(i,j,k , IB)) +
	       SQR(Udata(i,j,k , IC)) );
    } else {
      Udata(i,j,k , ID) = blast_density_out;
      Udata(i,j,k , IU) = 0.0;
      Udata(i,j,k , IV) = 0.0;
      Udata(i,j,k , IW) = 0.0;
      Udata(i,j,k , IA) = 0.5;
      Udata(i,j,k , IB) = 0.5;
      Udata(i,j,k , IC) = 0.5;
      Udata(i,j,k , IP) = blast_pressure_out/(gamma0-1.0) +
	0.5* ( SQR(Udata(i,j,k , IA)) +
	       SQR(Udata(i,j,k , IB)) +
	       SQR(Udata(i,j,k , IC)) );
    }
    
  } // end operator ()
  
  DataArray3d Udata;
  BlastParams bParams;
  
}; // InitBlastFunctor3D_MHD

/*************************************************/
/*************************************************/
/*************************************************/
enum OrszagTang_init_type {
  INIT_ALL_VAR_BUT_ENERGY = 0,
  INIT_ENERGY = 1
};
template<OrszagTang_init_type ot_type>
class InitOrszagTangFunctor3D : public MHDBaseFunctor3D {

public:
  InitOrszagTangFunctor3D(HydroParams params,
			  DataArray3d Udata) :
    MHDBaseFunctor3D(params), Udata(Udata)  {};
  
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
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    const double xmin = params.xmin;
    const double ymin = params.ymin;
    //const double zmin = params.zmin;
        
    const double dx = params.dx;
    const double dy = params.dy;
    //const double dz = params.dz;
    
    const real_t gamma0 = params.settings.gamma0;
    
    const double TwoPi = 4.0*asin(1.0);
    const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = gamma0/(2.0*TwoPi);
    const double d0    = gamma0*p0;
    const double v0    = 1.0;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    double xPos = xmin + dx/2 + (i-ghostWidth)*dx;
    double yPos = ymin + dy/2 + (j-ghostWidth)*dy;
    //double zPos = zmin + dz/2 + (k-ghostWidth)*dz;
    
    // density
    Udata(i,j,k,ID) = d0;
    
    // rho*vx
    Udata(i,j,k,IU)  = static_cast<real_t>(-d0*v0*sin(yPos*TwoPi));
    
    // rho*vy
    Udata(i,j,k,IV)  = static_cast<real_t>( d0*v0*sin(xPos*TwoPi));
    
    // rho*vz
    Udata(i,j,k,IW) =  ZERO_F;

    // bx, by, bz
    Udata(i,j,k, IBX) = -B0*sin(    yPos*TwoPi);
    Udata(i,j,k, IBY) =  B0*sin(2.0*xPos*TwoPi);
    Udata(i,j,k, IBZ) =  0.0;

  } // init_all_var_but_energy

  KOKKOS_INLINE_FUNCTION
  void init_energy(const int index) const
  {

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    //const double xmin = params.xmin;
    //const double ymin = params.ymin;
    //const double zmin = params.zmin;
    
    //const double dx = params.dx;
    //const double dy = params.dy;
    //const double dz = params.dz;
    
    const real_t gamma0 = params.settings.gamma0;
    
    const double TwoPi = 4.0*asin(1.0);
    const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = gamma0/(2.0*TwoPi);
    //const double d0    = gamma0*p0;
    //const double v0    = 1.0;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    //double xPos = xmin + dx/2 + (i-ghostWidth)*dx;
    //double yPos = ymin + dy/2 + (j-ghostWidth)*dy;
    //double zPos = zmin + dz/2 + (k-ghostWidth)*dz;
        
    if (i<isize-1 and j<jsize-1) {

      Udata(i,j,k,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,k,IU)) / Udata(i,j,k,ID) +
		SQR(Udata(i,j,k,IV)) / Udata(i,j,k,ID) +
		0.25*SQR(Udata(i,j,k,IBX) + Udata(i+1,j  ,k  ,IBX)) + 
		0.25*SQR(Udata(i,j,k,IBY) + Udata(i  ,j+1,k  ,IBY)) );

    } else if ( (i <isize-1) and (j==jsize-1)) {

      Udata(i,j,k,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,k,IU)) / Udata(i,j,k,ID) +
		SQR(Udata(i,j,k,IV)) / Udata(i,j,k,ID) +
		0.25*SQR(Udata(i,j,k,IBX) + Udata(i+1,j           ,k,IBX)) + 
		0.25*SQR(Udata(i,j,k,IBY) + Udata(i  ,2*ghostWidth,k,IBY)) );

    } else if ( (i==isize-1) and (j <jsize-1)) {

      Udata(i,j,k,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,k,IU)) / Udata(i,j,k,ID) +
		SQR(Udata(i,j,k,IV)) / Udata(i,j,k,ID) +
		0.25*SQR(Udata(i,j,k,IBX) + Udata(2*ghostWidth,j  ,k  ,IBX)) + 
		0.25*SQR(Udata(i,j,k,IBY) + Udata(i           ,j+1,k  ,IBY)) );

    } else if ( (i==isize-1) and (j==jsize-1) ) {

      Udata(i,j,k,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(i,j,k,IU)) / Udata(i,j,k,ID) +
		SQR(Udata(i,j,k,IV)) / Udata(i,j,k,ID) +
		0.25*SQR(Udata(i,j,k,IBX) + Udata(2*ghostWidth,j,k ,IBX)) + 
		0.25*SQR(Udata(i,j,k,IBY) + Udata(i,2*ghostWidth,k ,IBY)) );

    }
    
  } // init_energy
  
  DataArray3d Udata;
  
}; // InitOrszagTangFunctor3D

} // namespace muscl
} // namespace ppkMHD

#endif // MHD_RUN_FUNCTORS_3D_H_

