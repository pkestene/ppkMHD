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

namespace ppkMHD { namespace muscl { namespace mhd3d {

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor : public MHDBaseFunctor3D {

public:
  
  ComputeDtFunctor(HydroParams params,
		   DataArray Qdata) :
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
      qLoc.d  = Qdata(index,ID);
      qLoc.p  = Qdata(index,IP);
      qLoc.u  = Qdata(index,IU);
      qLoc.v  = Qdata(index,IV);
      qLoc.w  = Qdata(index,IW);
      qLoc.bx = Qdata(index,IBX);
      qLoc.by = Qdata(index,IBY);
      qLoc.bz = Qdata(index,IBZ);

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

  DataArray Qdata;
  
}; // ComputeDtFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ConvertToPrimitivesFunctor : public MHDBaseFunctor3D {

public:

  ConvertToPrimitivesFunctor(HydroParams params,
			     DataArray Udata,
			     DataArray Qdata) :
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

    // index of a neighbor cell
    int indexN;

    // magnetic field in neighbor cells
    real_t magFieldNeighbors[3];
    
    if(k >= 0 && k < ksize-1  &&
       j >= 0 && j < jsize-1  &&
       i >= 0 && i < isize-1 ) {
      
      MHDState uLoc; // conservative    variables in current cell
      MHDState qLoc; // primitive    variables in current cell
      real_t c;
      
      // get local conservative variable
      uLoc.d  = Udata(index,ID);
      uLoc.p  = Udata(index,IP);
      uLoc.u  = Udata(index,IU);
      uLoc.v  = Udata(index,IV);
      uLoc.w  = Udata(index,IW);
      uLoc.bx = Udata(index,IBX);
      uLoc.by = Udata(index,IBY);
      uLoc.bz = Udata(index,IBZ);

      // get mag field in neighbor cells
      indexN = coord2index(i+1,j  ,k  ,isize,jsize,ksize);
      magFieldNeighbors[IX] = Udata(indexN,IBX);
      indexN = coord2index(i  ,j+1,k  ,isize,jsize,ksize);
      magFieldNeighbors[IY] = Udata(indexN,IBY);
      indexN = coord2index(i  ,j  ,k+1,isize,jsize,ksize);
      magFieldNeighbors[IZ] = Udata(indexN,IBZ);
      
      // get primitive variables in current cell
      constoprim_mhd(uLoc, magFieldNeighbors, c, qLoc);

      // copy q state in q global
      Qdata(index,ID)  = qLoc.d;
      Qdata(index,IP)  = qLoc.p;
      Qdata(index,IU)  = qLoc.u;
      Qdata(index,IV)  = qLoc.v;
      Qdata(index,IW)  = qLoc.w;
      Qdata(index,IBX) = qLoc.bx;
      Qdata(index,IBY) = qLoc.by;
      Qdata(index,IBZ) = qLoc.bz;
      
    }
    
  }
  
  DataArray Udata;
  DataArray Qdata;
    
}; // ConvertToPrimitivesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeElecFieldFunctor : public MHDBaseFunctor3D {

public:

  ComputeElecFieldFunctor(HydroParams params,
			  DataArray Udata,
			  DataArray Qdata,
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
      v = ONE_FOURTH_F * ( Qdata   (coord2index(i  ,j-1,k-1,isize,jsize,ksize), IV) +
			   Qdata   (coord2index(i  ,j-1,k  ,isize,jsize,ksize), IV) +
			   Qdata   (coord2index(i  ,j  ,k-1,isize,jsize,ksize), IV) +
			   Qdata   (coord2index(i  ,j  ,k  ,isize,jsize,ksize), IV) );
      
      w = ONE_FOURTH_F * ( Qdata   (coord2index(i  ,j-1,k-1,isize,jsize,ksize), IW) +
			   Qdata   (coord2index(i  ,j-1,k  ,isize,jsize,ksize), IW) +
			   Qdata   (coord2index(i  ,j  ,k-1,isize,jsize,ksize), IW) +
			   Qdata   (coord2index(i  ,j  ,k  ,isize,jsize,ksize), IW) );
      
      B = HALF_F  * ( Udata(coord2index(i  ,j  ,k-1,isize,jsize,ksize),IB) +
		      Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize),IB) );
      
      C = HALF_F  * ( Udata(coord2index(i  ,j-1,k  ,isize,jsize,ksize),IC) +
		      Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize),IC) );
      
      ElecField(index,IX) = v*C-w*B;
      
      // compute Ey
      u = ONE_FOURTH_F * ( Qdata   (coord2index(i-1,j  ,k-1,isize,jsize,ksize),IU) +
			   Qdata   (coord2index(i-1,j  ,k  ,isize,jsize,ksize),IU) +
			   Qdata   (coord2index(i  ,j  ,k-1,isize,jsize,ksize),IU) +
			   Qdata   (coord2index(i  ,j  ,k  ,isize,jsize,ksize),IU) );
      
      w = ONE_FOURTH_F * ( Qdata   (coord2index(i-1,j  ,k-1,isize,jsize,ksize),IW) +
			   Qdata   (coord2index(i-1,j  ,k  ,isize,jsize,ksize),IW) +
			   Qdata   (coord2index(i  ,j  ,k-1,isize,jsize,ksize),IW) +
			   Qdata   (coord2index(i  ,j  ,k  ,isize,jsize,ksize),IW) );
      
      A = HALF_F  * ( Udata(coord2index(i  ,j  ,k-1,isize,jsize,ksize),IA) +
		      Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize),IA) );
      
      C = HALF_F  * ( Udata(coord2index(i-1,j  ,k  ,isize,jsize,ksize),IC) +
		      Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize),IC) );
      
      ElecField(index,IY) = w*A-u*C;
      
      // compute Ez
      u = ONE_FOURTH_F * ( Qdata   (coord2index(i-1,j-1,k  ,isize,jsize,ksize),IU) +
			   Qdata   (coord2index(i-1,j  ,k  ,isize,jsize,ksize),IU) +
			   Qdata   (coord2index(i  ,j-1,k  ,isize,jsize,ksize),IU) +
			   Qdata   (coord2index(i  ,j  ,k  ,isize,jsize,ksize),IU) );
      
      v = ONE_FOURTH_F * ( Qdata   (coord2index(i-1,j-1,k  ,isize,jsize,ksize),IV) +
			   Qdata   (coord2index(i-1,j  ,k  ,isize,jsize,ksize),IV) +
			   Qdata   (coord2index(i  ,j-1,k  ,isize,jsize,ksize),IV) +
			   Qdata   (coord2index(i  ,j  ,k  ,isize,jsize,ksize),IV) );
      
      A = HALF_F  * ( Udata(coord2index(i  ,j-1,k  ,isize,jsize,ksize),IA) +
		      Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize),IA) );
      
      B = HALF_F  * ( Udata(coord2index(i-1,j  ,k  ,isize,jsize,ksize),IB) +
		      Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize),IB) );
      
      ElecField(index,IZ) = u*B-v*A;
      
    }
  } // operator ()

  DataArray Udata;
  DataArray Qdata;  
  DataArrayVector3 ElecField;
  
}; // ComputeElecFieldFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeMagSlopesFunctor : public MHDBaseFunctor3D {

public:

  ComputeMagSlopesFunctor(HydroParams      params,
			  DataArray        Udata,
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
      bfSlopes[0]  = Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IA);
      bfSlopes[1]  = Udata(coord2index(i  ,j+1,k  ,isize,jsize,ksize), IA);
      bfSlopes[2]  = Udata(coord2index(i  ,j-1,k  ,isize,jsize,ksize), IA);
      bfSlopes[3]  = Udata(coord2index(i  ,j  ,k+1,isize,jsize,ksize), IA);
      bfSlopes[4]  = Udata(coord2index(i  ,j  ,k-1,isize,jsize,ksize), IA);
      
      bfSlopes[5]  = Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IB);
      bfSlopes[6]  = Udata(coord2index(i+1,j  ,k  ,isize,jsize,ksize), IB);
      bfSlopes[7]  = Udata(coord2index(i-1,j  ,k  ,isize,jsize,ksize), IB);
      bfSlopes[8]  = Udata(coord2index(i  ,j  ,k+1,isize,jsize,ksize), IB);
      bfSlopes[9]  = Udata(coord2index(i  ,j  ,k-1,isize,jsize,ksize), IB);
      
      bfSlopes[10] = Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IC);
      bfSlopes[11] = Udata(coord2index(i+1,j  ,k  ,isize,jsize,ksize), IC);
      bfSlopes[12] = Udata(coord2index(i-1,j  ,k  ,isize,jsize,ksize), IC);
      bfSlopes[13] = Udata(coord2index(i  ,j+1,k  ,isize,jsize,ksize), IC);
      bfSlopes[14] = Udata(coord2index(i  ,j-1,k  ,isize,jsize,ksize), IC);
      
      // compute magnetic slopes
      slope_unsplit_mhd_3d(bfSlopes, dbfSlopes);
      
      // store magnetic slopes
      DeltaA(index,0) = dbfX[IX];
      DeltaA(index,1) = dbfY[IX];
      DeltaA(index,2) = dbfZ[IX];
      
      DeltaB(index,0) = dbfX[IY];
      DeltaB(index,1) = dbfY[IY];
      DeltaB(index,2) = dbfZ[IY];
      
      DeltaC(index,0) = dbfX[IZ];
      DeltaC(index,1) = dbfY[IZ];
      DeltaC(index,2) = dbfZ[IZ];
      
    }

  } // operator ()
  DataArray Udata;
  DataArrayVector3 DeltaA;
  DataArrayVector3 DeltaB;
  DataArrayVector3 DeltaC;
    
}; // class ComputeMagSlopesFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeTraceFunctor : public MHDBaseFunctor3D {

public:

  ComputeTraceFunctor(HydroParams params,
		      DataArray Udata,
		      DataArray Qdata,
		      DataArrayVector3 DeltaA,
		      DataArrayVector3 DeltaB,
		      DataArrayVector3 DeltaC,
		      DataArrayVector3 ElecField,
		      DataArray Qm_x,
		      DataArray Qm_y,
		      DataArray Qm_z,
		      DataArray Qp_x,
		      DataArray Qp_y,
		      DataArray Qp_z,
		      DataArray QEdge_RT,
		      DataArray QEdge_RB,
		      DataArray QEdge_LT,
		      DataArray QEdge_LB,
		      DataArray QEdge_RT2,
		      DataArray QEdge_RB2,
		      DataArray QEdge_LT2,
		      DataArray QEdge_LB2,
		      DataArray QEdge_RT3,
		      DataArray QEdge_RB3,
		      DataArray QEdge_LT3,
		      DataArray QEdge_LB3,
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
      get_state(Qdata, coord2index(i  ,j  ,k  , isize, jsize, ksize), q      );
      get_state(Qdata, coord2index(i+1,j  ,k  , isize, jsize, ksize), qPlusX );
      get_state(Qdata, coord2index(i-1,j  ,k  , isize, jsize, ksize), qMinusX);
      get_state(Qdata, coord2index(i  ,j+1,k  , isize, jsize, ksize), qPlusY );
      get_state(Qdata, coord2index(i  ,j-1,k  , isize, jsize, ksize), qMinusY);
      get_state(Qdata, coord2index(i  ,j  ,k+1, isize, jsize, ksize), qPlusZ );
      get_state(Qdata, coord2index(i  ,j  ,k-1, isize, jsize, ksize), qMinusZ);
      
      // get hydro slopes dq
      slope_unsplit_hydro_3d(q, 
			     qPlusX, qMinusX, 
			     qPlusY, qMinusY, 
			     qPlusZ, qMinusZ,
			     dq);
      
      // get face-centered magnetic components
      bfNb[0] = Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IA);
      bfNb[1] = Udata(coord2index(i+1,j  ,k  ,isize,jsize,ksize), IA);
      bfNb[2] = Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IB);
      bfNb[3] = Udata(coord2index(i  ,j+1,k  ,isize,jsize,ksize), IB);
      bfNb[4] = Udata(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IC);
      bfNb[5] = Udata(coord2index(i  ,j  ,k+1,isize,jsize,ksize), IC);
      
      // get dbf (transverse magnetic slopes)
      dbf[0]  = DeltaA(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IY);
      dbf[1]  = DeltaA(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IZ);
      dbf[2]  = DeltaB(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IX);
      dbf[3]  = DeltaB(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IZ);
      dbf[4]  = DeltaC(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IX);
      dbf[5]  = DeltaC(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IY);
      
      dbf[6]  = DeltaA(coord2index(i+1,j  ,k  ,isize,jsize,ksize), IY);
      dbf[7]  = DeltaA(coord2index(i+1,j  ,k  ,isize,jsize,ksize), IZ);
      dbf[8]  = DeltaB(coord2index(i  ,j+1,k  ,isize,jsize,ksize), IX);
      dbf[9]  = DeltaB(coord2index(i  ,j+1,k  ,isize,jsize,ksize), IZ);
      dbf[10] = DeltaC(coord2index(i  ,j  ,k+1,isize,jsize,ksize), IX);
      dbf[11] = DeltaC(coord2index(i  ,j  ,k+1,isize,jsize,ksize), IY);
      
      // get electric field components
      Ex[0][0] = ElecField(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IX);
      Ex[0][1] = ElecField(coord2index(i  ,j  ,k+1,isize,jsize,ksize), IX);
      Ex[1][0] = ElecField(coord2index(i  ,j+1,k  ,isize,jsize,ksize), IX);
      Ex[1][1] = ElecField(coord2index(i  ,j+1,k+1,isize,jsize,ksize), IX);
      
      Ey[0][0] = ElecField(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IY);
      Ey[0][1] = ElecField(coord2index(i  ,j  ,k+1,isize,jsize,ksize), IY);
      Ey[1][0] = ElecField(coord2index(i+1,j  ,k  ,isize,jsize,ksize), IY);
      Ey[1][1] = ElecField(coord2index(i+1,j  ,k+1,isize,jsize,ksize), IY);
      
      Ez[0][0] = ElecField(coord2index(i  ,j  ,k  ,isize,jsize,ksize), IZ);
      Ez[0][1] = ElecField(coord2index(i  ,j+1,k  ,isize,jsize,ksize), IZ);
      Ez[1][0] = ElecField(coord2index(i+1,j  ,k  ,isize,jsize,ksize), IZ);
      Ez[1][1] = ElecField(coord2index(i+1,j+1,k  ,isize,jsize,ksize), IZ);
      
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
      set_state(Qm_x, index, qm[0]);
      set_state(Qp_x, index, qp[0]);
      set_state(Qm_y, index, qm[1]);
      set_state(Qp_y, index, qp[1]);
      set_state(Qm_z, index, qm[2]);
      set_state(Qp_z, index, qp[2]);
      
      set_state(QEdge_RT , index, qEdge[IRT][0]); 
      set_state(QEdge_RB , index, qEdge[IRB][0]); 
      set_state(QEdge_LT , index, qEdge[ILT][0]); 
      set_state(QEdge_LB , index, qEdge[ILB][0]); 
      
      set_state(QEdge_RT2, index, qEdge[IRT][1]); 
      set_state(QEdge_RB2, index, qEdge[IRB][1]); 
      set_state(QEdge_LT2, index, qEdge[ILT][1]); 
      set_state(QEdge_LB2, index, qEdge[ILB][1]); 
      
      set_state(QEdge_RT3, index, qEdge[IRT][2]); 
      set_state(QEdge_RB3, index, qEdge[IRB][2]); 
      set_state(QEdge_LT3, index, qEdge[ILT][2]); 
      set_state(QEdge_LB3, index, qEdge[ILB][2]); 
      
    }
    
  } // operator ()
      
  DataArray Udata, Qdata;
  DataArrayVector3 DeltaA, DeltaB, DeltaC, ElecField;
  DataArray Qm_x, Qm_y, Qm_z;
  DataArray Qp_x, Qp_y, Qp_z;
  DataArray QEdge_RT,  QEdge_RB,  QEdge_LT,  QEdge_LB;
  DataArray QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  real_t dtdx, dtdy, dtdz;
  
}; // class ComputeTraceFunctor
  
/*************************************************/
/*************************************************/
/*************************************************/
class ComputeFluxesAndStoreFunctor : public MHDBaseFunctor3D {

public:

  ComputeFluxesAndStoreFunctor(HydroParams params,
			       DataArray Qm_x,
			       DataArray Qm_y,
			       DataArray Qm_z,
			       DataArray Qp_x,
			       DataArray Qp_y,
			       DataArray Qp_z,
			       DataArray Fluxes_x,
			       DataArray Fluxes_y,
			       DataArray Fluxes_z,
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

      int index2;

      //
      // Solve Riemann problem at X-interfaces and compute X-fluxes
      //
      index2 = coord2index(i-1,j  ,k  ,isize,jsize,ksize);
      get_state(Qm_x, index2, qleft);
      
      get_state(Qp_x, index, qright);
      
      // compute hydro flux along X
      riemann_hlld(qleft,qright,flux);

      // store fluxes
      set_state(Fluxes_x, index, flux);

      //
      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      //
      index2 = coord2index(i  ,j-1,k  ,isize,jsize,ksize);
      get_state(Qm_y, index2, qleft);
      swapValues(&(qleft.u)  ,&(qleft.v) );
      swapValues(&(qleft.bx) ,&(qleft.by) );

      get_state(Qp_y, index, qright);
      swapValues(&(qright.u)  ,&(qright.v) );
      swapValues(&(qright.bx) ,&(qright.by) );
      
      // compute hydro flux along Y
      riemann_hlld(qleft,qright,flux);
            
      // store fluxes
      set_state(Fluxes_y, index, flux);
      
      //
      // Solve Riemann problem at Z-interfaces and compute Z-fluxes
      //
      index2 = coord2index(i  ,j  ,k-1,isize,jsize,ksize);
      get_state(Qm_z, index2, qleft);
      swapValues(&(qleft.u)  ,&(qleft.w) );
      swapValues(&(qleft.bx) ,&(qleft.bz) );

      get_state(Qp_z, index, qright);
      swapValues(&(qright.u)  ,&(qright.w) );
      swapValues(&(qright.bx) ,&(qright.bz) );
      
      // compute hydro flux along Z
      riemann_hlld(qleft,qright,flux);
            
      // store fluxes
      set_state(Fluxes_z, index, flux);

    }
    
  }
  
  DataArray Qm_x, Qm_y, Qm_z;
  DataArray Qp_x, Qp_y, Qp_z;
  DataArray Fluxes_x, Fluxes_y, Fluxes_z;
  real_t dtdx, dtdy, dtdz;
  
}; // ComputeFluxesAndStoreFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeEmfAndStoreFunctor : public MHDBaseFunctor3D {

public:

  ComputeEmfAndStoreFunctor(HydroParams params,
			    DataArray QEdge_RT,
			    DataArray QEdge_RB,
			    DataArray QEdge_LT,
			    DataArray QEdge_LB,
			    DataArray QEdge_RT2,
			    DataArray QEdge_RB2,
			    DataArray QEdge_LT2,
			    DataArray QEdge_LB2,
			    DataArray QEdge_RT3,
			    DataArray QEdge_RB3,
			    DataArray QEdge_LT3,
			    DataArray QEdge_LB3,
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
      get_state(QEdge_RT3, coord2index(i-1,j-1,k  ,isize, jsize, ksize), qEdge_emf[IRT]);
      get_state(QEdge_RB3, coord2index(i-1,j  ,k  ,isize, jsize, ksize), qEdge_emf[IRB]); 
      get_state(QEdge_LT3, coord2index(i  ,j-1,k  ,isize, jsize, ksize), qEdge_emf[ILT]);
      get_state(QEdge_LB3, coord2index(i  ,j  ,k  ,isize, jsize, ksize), qEdge_emf[ILB]);

      Emf(index,I_EMFZ) = compute_emf<EMFZ>(qEdge_emf);
      
      // actually compute emfY (take care that RB and LT are
      // swapped !!!)
      get_state(QEdge_RT2, coord2index(i-1,j  ,k-1,isize, jsize, ksize), qEdge_emf[IRT]);
      get_state(QEdge_LT2, coord2index(i  ,j  ,k-1,isize, jsize, ksize), qEdge_emf[IRB]); 
      get_state(QEdge_RB2, coord2index(i-1,j  ,k  ,isize, jsize, ksize), qEdge_emf[ILT]);
      get_state(QEdge_LB2, coord2index(i  ,j  ,k  ,isize, jsize, ksize), qEdge_emf[ILB]);

      Emf(index,I_EMFY) = compute_emf<EMFY>(qEdge_emf);
      
      // actually compute emfX
      get_state(QEdge_RT, coord2index(i  ,j-1,k-1,isize, jsize, ksize), qEdge_emf[IRT]);
      get_state(QEdge_RB, coord2index(i  ,j-1,k  ,isize, jsize, ksize), qEdge_emf[IRB]); 
      get_state(QEdge_LT, coord2index(i  ,j  ,k-1,isize, jsize, ksize), qEdge_emf[ILT]);
      get_state(QEdge_LB, coord2index(i  ,j  ,k  ,isize, jsize, ksize), qEdge_emf[ILB]);

      Emf(index,I_EMFX) = compute_emf<EMFX>(qEdge_emf);
    }
  }

  DataArray QEdge_RT,  QEdge_RB,  QEdge_LT,  QEdge_LB;
  DataArray QEdge_RT2, QEdge_RB2, QEdge_LT2, QEdge_LB2;
  DataArray QEdge_RT3, QEdge_RB3, QEdge_LT3, QEdge_LB3;
  DataArrayVector3 Emf;
  real_t dtdx, dtdy, dtdz;

}; // ComputeEmfAndStoreFunctor

  
/*************************************************/
/*************************************************/
/*************************************************/
class UpdateFunctor : public MHDBaseFunctor3D {

public:

  UpdateFunctor(HydroParams params,
		DataArray Udata,
		DataArray FluxData_x,
		DataArray FluxData_y,
		DataArray FluxData_z,
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

    int index2;
    
    if(k >= ghostWidth && k < ksize-ghostWidth  &&
       j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      MHDState udata;
      MHDState flux;
      get_state(Udata, index, udata);

      // add up contributions from all 6 faces
      
      get_state(FluxData_x, index, flux);      
      udata.d  +=  flux.d*dtdx;
      udata.p  +=  flux.p*dtdx;
      udata.u  +=  flux.u*dtdx;
      udata.v  +=  flux.v*dtdx;
      udata.w  +=  flux.w*dtdx;
      
      index2 = coord2index(i+1,j  ,k  ,isize,jsize,ksize);
      get_state(FluxData_x, index2, flux);
      udata.d  -=  flux.d*dtdx;
      udata.p  -=  flux.p*dtdx;
      udata.u  -=  flux.u*dtdx;
      udata.v  -=  flux.v*dtdx;
      udata.w  -=  flux.w*dtdx;
      
      get_state(FluxData_y, index, flux);
      udata.d  +=  flux.d*dtdy;
      udata.p  +=  flux.p*dtdy;
      udata.u  +=  flux.v*dtdy; //
      udata.v  +=  flux.u*dtdy; //
      udata.w  +=  flux.w*dtdy;
                  
      index2 = coord2index(i  ,j+1,k  ,isize,jsize,ksize);
      get_state(FluxData_y, index2, flux);
      udata.d  -=  flux.d*dtdy;
      udata.p  -=  flux.p*dtdy;
      udata.u  -=  flux.v*dtdy; //
      udata.v  -=  flux.u*dtdy; //
      udata.w  -=  flux.w*dtdy;

      get_state(FluxData_z, index, flux);
      udata.d  +=  flux.d*dtdy;
      udata.p  +=  flux.p*dtdy;
      udata.u  +=  flux.w*dtdy; //
      udata.v  +=  flux.v*dtdy;
      udata.w  +=  flux.u*dtdy; //

      index2 = coord2index(i  ,j  ,k+1,isize,jsize,ksize);
      get_state(FluxData_z, index2, flux);
      udata.d  -=  flux.d*dtdz;
      udata.p  -=  flux.p*dtdz;
      udata.u  -=  flux.w*dtdz; //
      udata.v  -=  flux.v*dtdz;
      udata.w  -=  flux.u*dtdz; //
      
      // write back result in Udata
      set_state(Udata, index, udata);
      
    } // end if
    
  } // end operator ()
  
  DataArray Udata;
  DataArray FluxData_x, FluxData_y, FluxData_z;
  real_t dtdx, dtdy, dtdz;
  
}; // UpdateFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class UpdateEmfFunctor : public MHDBaseFunctor3D {

public:

  UpdateEmfFunctor(HydroParams params,
		   DataArray Udata,
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
      get_state(Udata, index, udata);

      if (k<ksize-ghostWidth) {
	udata.bx += ( Emf(coord2index(i  ,j+1, k, isize, jsize, ksize), I_EMFZ) - 
		      Emf(coord2index(i,  j  , k, isize, jsize, ksize), I_EMFZ) ) * dtdy;
	
	udata.by -= ( Emf(coord2index(i+1,j  , k, isize, jsize, ksize), I_EMFZ) - 
		      Emf(coord2index(i  ,j  , k, isize, jsize, ksize), I_EMFZ) ) * dtdx;
	
      }
      
      // update BX
      udata.bx -= ( Emf(coord2index(i,j,k+1, isize, jsize, ksize), I_EMFY) -
		    Emf(coord2index(i,j,k  , isize, jsize, ksize), I_EMFY) ) * dtdz;
      
      // update BY
      udata.by += ( Emf(coord2index(i,j,k+1, isize, jsize, ksize), I_EMFX) -
		    Emf(coord2index(i,j,k  , isize, jsize, ksize), I_EMFX) ) * dtdz;
      
      // update BZ
      udata.bz += ( Emf(coord2index(i+1,j  ,k, isize, jsize, ksize), I_EMFY) -
		    Emf(coord2index(i  ,j  ,k, isize, jsize, ksize), I_EMFY) ) * dtdx;
      
      udata.bz -= ( Emf(coord2index(i  ,j+1,k, isize, jsize, ksize), I_EMFX) -
		    Emf(coord2index(i  ,j  ,k, isize, jsize, ksize), I_EMFX) ) * dtdy;

      Udata(index, IA) = udata.bx;
      Udata(index, IB) = udata.by;
      Udata(index, IC) = udata.bz;
      
    }
  } // operator()

  DataArray Udata;
  DataArrayVector3 Emf;
  real_t dtdx, dtdy, dtdz;

}; // UpdateEmfFunctor


/*************************************************/
/*************************************************/
/*************************************************/
class InitImplodeFunctor : public MHDBaseFunctor3D {

public:
  InitImplodeFunctor(HydroParams params,
		     DataArray Udata) :
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
      Udata(index , ID)  = 1.0;
      Udata(index , IU)  = 0.0;
      Udata(index , IV)  = 0.0;
      Udata(index , IW)  = 0.0;
      Udata(index , IBX) = 0.5;
      Udata(index , IBY) = 0.0;
      Udata(index , IBZ) = 0.0;
      Udata(index , IP)  = 1.0/(gamma0-1.0) +
	0.5* ( SQR(Udata(index , IBX)) +
	       SQR(Udata(index , IBY)) +
	       SQR(Udata(index , IBZ)) );
    } else {
      Udata(index , ID)  = 0.125;
      Udata(index , IU)  = 0.0;
      Udata(index , IV)  = 0.0;
      Udata(index , IW)  = 0.0;
      Udata(index , IBX) = 0.5;
      Udata(index , IBY) = 0.0;
      Udata(index , IBZ) = 0.0;
      Udata(index , IP)  = 0.14/(gamma0-1.0)  +
	0.5* ( SQR(Udata(index , IBX)) +
	       SQR(Udata(index , IBY)) +
	       SQR(Udata(index , IBZ)) );
    }
    
  } // end operator ()

  DataArray Udata;

}; // InitImplodeFunctor

/*************************************************/
/*************************************************/
/*************************************************/
class InitBlastFunctor : public MHDBaseFunctor3D {

public:
  InitBlastFunctor(HydroParams params,
		   BlastParams bParams,
		   DataArray Udata) :
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
      Udata(index , ID) = blast_density_in;
      Udata(index , IU) = 0.0;
      Udata(index , IV) = 0.0;
      Udata(index , IW) = 0.0;
      Udata(index , IA) = 0.5;
      Udata(index , IB) = 0.5;
      Udata(index , IC) = 0.5;
      Udata(index , IP) = blast_pressure_in/(gamma0-1.0) +
	0.5* ( SQR(Udata(index , IA)) +
	       SQR(Udata(index , IB)) +
	       SQR(Udata(index , IC)) );
    } else {
      Udata(index , ID) = blast_density_out;
      Udata(index , IU) = 0.0;
      Udata(index , IV) = 0.0;
      Udata(index , IW) = 0.0;
      Udata(index , IA) = 0.5;
      Udata(index , IB) = 0.5;
      Udata(index , IC) = 0.5;
      Udata(index , IP) = blast_pressure_out/(gamma0-1.0) +
	0.5* ( SQR(Udata(index , IA)) +
	       SQR(Udata(index , IB)) +
	       SQR(Udata(index , IC)) );
    }
    
  } // end operator ()
  
  DataArray Udata;
  BlastParams bParams;
  
}; // InitBlastFunctor

/*************************************************/
/*************************************************/
/*************************************************/
enum OrszagTang_init_type {
  INIT_ALL_VAR_BUT_ENERGY = 0,
  INIT_ENERGY = 1
};
template<OrszagTang_init_type ot_type>
class InitOrszagTangFunctor : public MHDBaseFunctor3D {

public:
  InitOrszagTangFunctor(HydroParams params,
			DataArray Udata) :
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
    Udata(index,ID) = d0;
    
    // rho*vx
    Udata(index,IU)  = static_cast<real_t>(-d0*v0*sin(yPos*TwoPi));
    
    // rho*vy
    Udata(index,IV)  = static_cast<real_t>( d0*v0*sin(xPos*TwoPi));
    
    // rho*vz
    Udata(index,IW) =  ZERO_F;

    // bx, by, bz
    Udata(index, IBX) = -B0*sin(    yPos*TwoPi);
    Udata(index, IBY) =  B0*sin(2.0*xPos*TwoPi);
    Udata(index, IBZ) =  0.0;

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
    
    int index_ip1 = coord2index(i+1,j  ,k  ,isize,jsize,ksize);
    int index_jp1 = coord2index(i  ,j+1,k  ,isize,jsize,ksize);
    int index_kp1 = coord2index(i  ,j  ,k+1,isize,jsize,ksize);

    int index_igk  = coord2index(i,2*ghostWidth,k,isize,jsize,ksize);
    int index_gjk  = coord2index(2*ghostWidth,j,k,isize,jsize,ksize);
    int index_ijg  = coord2index(i,j,2*ghostWidth,isize,jsize,ksize);
    
    if (i<isize-1 and j<jsize-1) {

      Udata(index,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(index,IU)) / Udata(index,ID) +
		SQR(Udata(index,IV)) / Udata(index,ID) +
		0.25*SQR(Udata(index,IBX) + Udata(index_ip1,IBX)) + 
		0.25*SQR(Udata(index,IBY) + Udata(index_jp1,IBY)) );

    } else if ( (i <isize-1) and (j==jsize-1)) {

      Udata(index,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(index,IU)) / Udata(index,ID) +
		SQR(Udata(index,IV)) / Udata(index,ID) +
		0.25*SQR(Udata(index,IBX) + Udata(index_ip1,IBX)) + 
		0.25*SQR(Udata(index,IBY) + Udata(index_igk,IBY)) );

    } else if ( (i==isize-1) and (j <jsize-1)) {

      Udata(index,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(index,IU)) / Udata(index,ID) +
		SQR(Udata(index,IV)) / Udata(index,ID) +
		0.25*SQR(Udata(index,IBX) + Udata(index_gjk ,IBX)) + 
		0.25*SQR(Udata(index,IBY) + Udata(index_jp1,IBY)) );

    } else if ( (i==isize-1) and (j==jsize-1) ) {

      Udata(index,IP)  = p0 / (gamma0-1.0) +
	0.5 * ( SQR(Udata(index,IU)) / Udata(index,ID) +
		SQR(Udata(index,IV)) / Udata(index,ID) +
		0.25*SQR(Udata(index,IBX) + Udata(index_gjk ,IBX)) + 
		0.25*SQR(Udata(index,IBY) + Udata(index_igk ,IBY)) );

    }
    
  } // init_energy
  
  DataArray Udata;
  
}; // InitOrszagTangFunctor

/*************************************************/
/*************************************************/
/*************************************************/
template <FaceIdType faceId>
class MakeBoundariesFunctor : public MHDBaseFunctor3D {

public:

  MakeBoundariesFunctor(HydroParams params,
			DataArray Udata) :
    MHDBaseFunctor3D(params), Udata(Udata)  {};
  
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
	    if (iVar==IA) sign=-ONE_F;
	  } else if( boundary_type == BC_NEUMANN ) {
	    i0=ghostWidth;
	  } else { // periodic
	    i0=nx+i;
	  }

	  index_out = coord2index(i ,j,k,isize,jsize,ksize);
	  index_in  = coord2index(i0,j,k,isize,jsize,ksize);
	  Udata(index_out , iVar) = Udata(index_in , iVar)*sign;

	}

      }
    } // end FACE_XMIN

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
	    if (iVar==IA) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    i0=nx+ghostWidth-1;
	  } else { // periodic
	    i0=i-nx;
	  }

	  index_out = coord2index(i ,j,k,isize,jsize,ksize);
	  index_in  = coord2index(i0,j,k,isize,jsize,ksize);
	  Udata(index_out, iVar) = Udata(index_in, iVar)*sign;

	}
      }
    } // end FACE_XMAX
    
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
	    if (iVar==IB) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ghostWidth;
	  } else { // periodic
	    j0=ny+j;
	  }

	  index_out = coord2index(i,j ,k,isize,jsize,ksize);
	  index_in  = coord2index(i,j0,k,isize,jsize,ksize);
	  Udata(index_out, iVar) = Udata(index_in, iVar)*sign;
	
	}
      }
    } // end FACE_YMIN

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
	    if (iVar==IB) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    j0=ny+ghostWidth-1;
	  } else { // periodic
	    j0=j-ny;
	  }

	  index_out = coord2index(i,j ,k,isize,jsize,ksize);
	  index_in  = coord2index(i,j0,k,isize,jsize,ksize);
	  Udata(index_out , iVar) = Udata(index_in , iVar)*sign;

	}

      }
    } // end FACE_YMAX
    
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
	    if (iVar==IC) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    k0=ghostWidth;
	  } else { // periodic
	    k0=nz+k;
	  }

	  index_out = coord2index(i,j,k ,isize,jsize,ksize);
	  index_in  = coord2index(i,j,k0,isize,jsize,ksize);
	  Udata(index_out, iVar) = Udata(index_in, iVar)*sign;
	
	}
      }
    } // end FACE_ZMIN
    
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
	    if (iVar==IC) sign=-ONE_F;
	  } else if ( boundary_type == BC_NEUMANN ) {
	    k0=nz+ghostWidth-1;
	  } else { // periodic
	    k0=k-nz;
	  }

	  index_out = coord2index(i,j,k ,isize,jsize,ksize);
	  index_in  = coord2index(i,j,k0,isize,jsize,ksize);
	  Udata(index_out, iVar) = Udata(index_in, iVar)*sign;
	
	}
      }
    } // end FACE_ZMAX
    
  } // end operator ()

  DataArray Udata;
  
}; // MakeBoundariesFunctor

} // namespace mhd3d
} // namespace muscl
} // namespace ppkMHD

#endif // MHD_RUN_FUNCTORS_3D_H_

