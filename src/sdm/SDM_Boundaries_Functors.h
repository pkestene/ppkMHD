#ifndef SDM_BOUNDARIES_FUNCTORS_H_
#define SDM_BOUNDARIES_FUNCTORS_H_

#include "shared/HydroParams.h"    // for HydroParams
#include "shared/kokkos_shared.h"  // for Data arrays

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Functors to update ghost cells for SDM (Spectral Difference Method schemes).
 *
 * \tparam dim 2d or 3d
 * \tparam N order of the SDM scheme (also number of Dof per direction per cell)
 * \tparam faceId identifies on which face will the border condition be fixed
 *
 * At some point, we should consider BC like ASZ (Absorbing Sponge Zone) or PML
 * (Perfectly Matched Layer); see article
 * "Absorbing boundary conditions for the Euler and Navier–Stokes
 *  equations with the spectral difference method", by Ying Zhou, Z.J. Wang
 * Journal of Computational Physics, 229 (2010) 8733–8749
 */
template <int dim, int N, FaceIdType faceId>
class MakeBoundariesFunctor_SDM  : public SDMBaseFunctor<dim,N> {
  
public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  MakeBoundariesFunctor_SDM(HydroParams           params,
			    SDM_Geometry<dim,N>   sdm_geom,
			    DataArray             Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    Udata(Udata) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    DataArray           Udata)
  {
    const int ghostWidth=params.ghostWidth;
    int max_size = std::max(params.isize,params.jsize);
    int nbIter = ghostWidth * max_size;
    nbIter *= (N*N);

    if (dim==3) {
      max_size = std::max(max_size,params.ksize);
      nbIter = ghostWidth * max_size * max_size;
      nbIter *= (N*N*N);
    }

    MakeBoundariesFunctor_SDM<dim,N,faceId> functor(params, sdm_geom, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {
    const int nx = this->params.nx;
    const int ny = this->params.ny;
    
    const int ghostWidth = this->params.ghostWidth;
    const int NghostWidth = N*ghostWidth;
    const int nbvar = this->params.nbvar;
    
    const int imin = this->params.imin;
    const int imax = this->params.imax;
    
    const int jmin = this->params.jmin;
    const int jmax = this->params.jmax;

    // global index
    int ii,jj;

    // local index
    int i,j;
    int idx, idy;

    int boundary_type;
    
    int i0, j0, ii0, jj0;
    int iVar;

    if (faceId == FACE_XMIN) {
      
      // boundary xmin
      boundary_type = this->params.boundary_type_xmin;

      // compute global
      jj = index / NghostWidth;
      ii = index - jj*NghostWidth;

      // compute local
      global2local(ii,jj, i,j,idx,idy, N);

      if(j >= jmin and j <= jmax    and
	 i >= 0    and i <ghostWidth) {
        
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            i0  = 2*ghostWidth-1-i;
            ii0 = N*i0 + N-1-idx;

            if (iVar==IU) sign=-ONE_F;
            
            // mirror DoFs idx <-> N-1-idx
            Udata(ii,jj,iVar) = Udata(ii0,jj,iVar)*sign;
            
          } else if( boundary_type == BC_NEUMANN ) {
            
            // TO BE MODIFIED: ghost cell DoFs must be extrapolated from
            // the inside
            i0  = ghostWidth;
            ii0 = N*i0 + idx;
            Udata(ii,jj,iVar) = Udata(ii0,jj,iVar);
            
          } else { // periodic
            
            i0  = nx+i;
            ii0 = N*i0 + idx;

            Udata(ii,jj,iVar) = Udata(ii0,jj,iVar);
            
          }
	  
	  
        } // end for iVar
	
      } // end guard
      
    } // end FACE_XMIN

    if (faceId == FACE_XMAX) {
      
      // boundary xmax
      boundary_type = this->params.boundary_type_xmax;

      // compute global
      jj = index / NghostWidth;
      ii = index - jj*NghostWidth;
      ii += (nx+ghostWidth)*N;

      // compute local
      global2local(ii,jj, i,j,idx,idy, N);

      if(j >= jmin          and j <= jmax             and
	 i >= nx+ghostWidth and i <= nx+2*ghostWidth-1) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            i0  = 2*nx+2*ghostWidth-1-i;
            ii0 = N*i0 + N-1-idx;

            if (iVar==IU) sign=-ONE_F;
            
            // mirror DoFs idx <-> N-1-idx
            Udata(ii,jj,iVar) = Udata(ii0,jj,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            i0  = nx+ghostWidth-1;
            ii0 = N*i0 + idx;

            Udata(ii,jj,iVar) = Udata(ii0,jj,iVar);
            
          } else { // periodic
            
            i0  = i-nx;
            ii0 = N*i0 + idx;

            Udata(ii,jj,iVar) = Udata(ii0,jj,iVar);
          }
	  
        } // end for iVar

      } // end guard
      
    } // end FACE_XMAX
    
    if (faceId == FACE_YMIN) {
      
      // boundary ymin
      boundary_type = this->params.boundary_type_ymin;
      
      // compute global
      ii = index / NghostWidth;
      jj = index - ii*NghostWidth;

      // compute local
      global2local(ii,jj, i,j,idx,idy, N);

      if(i >= imin and i <= imax    and
	 j >= 0    and j <ghostWidth) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ghostWidth-1-j;
            jj0 = N*j0 + N-1-idy;

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(ii,jj,iVar) = Udata(ii,jj0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ghostWidth;
            jj0 = N*j0 + idy;

            Udata(ii,jj,iVar) = Udata(ii,jj0,iVar);
            
          } else { // periodic
            
            j0  = ny+j;
            jj0 = N*j0 + idy;

            Udata(ii,jj,iVar) = Udata(ii,jj0,iVar);
            
          }
	  
        } // end for iVar

      } // end guard
      
    } // end FACE_YMIN

    if (faceId == FACE_YMAX) {

      // boundary ymax
      boundary_type = this->params.boundary_type_ymax;

      // compute global
      ii = index / NghostWidth;
      jj = index - ii*NghostWidth;
      jj += (ny+ghostWidth)*N;

      // compute local
      global2local(ii,jj, i,j,idx,idy, N);

      if(i >= imin          and i <= imax              and
	 j >= ny+ghostWidth and j <= ny+2*ghostWidth-1) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ny+2*ghostWidth-1-j;
            jj0 = N*j0 + N-1-idy;

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(ii,jj,iVar) = Udata(ii,jj0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ny+ghostWidth-1;
            jj0 = N*j0 + idy;

            Udata(ii,jj,iVar) = Udata(ii,jj0,iVar);
            
          } else { // periodic
            
            j0  = j-ny;
            jj0 = N*j0 + idy;

            Udata(ii,jj,iVar) = Udata(ii,jj0,iVar);
            
          }
	  
        } // end for iVar

      } // end guard
      
    } // end FACE_YMAX
    
  } // end operator () - 2d

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int nx = this->params.nx;
    const int ny = this->params.ny;
    const int nz = this->params.nz;
    
    //const int isize = this->params.isize;
    //const int jsize = this->params.jsize;
    //const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    const int iisize = this->params.isize*N;
    const int jjsize = this->params.jsize*N;
    const int NghostWidth = this->params.ghostWidth*N;

    const int nbvar = this->params.nbvar;
    
    const int imin = this->params.imin;
    const int imax = this->params.imax;
    
    const int jmin = this->params.jmin;
    const int jmax = this->params.jmax;

    const int kmin = this->params.kmin;
    const int kmax = this->params.kmax;
    
    // global index
    int ii,jj,kk;

    // local index
    int i,j,k;
    int idx, idy, idz;
    
    int boundary_type;
    
    int i0, j0, k0;
    int ii0, jj0, kk0;
    int iVar;
    
    if (faceId == FACE_XMIN) {
      
      // boundary xmin (index = ii + jj * NghostWidth + kk * NghostWidth*jjsize)
      kk = index / (NghostWidth*jjsize);
      jj = (index - kk*NghostWidth*jjsize) / NghostWidth;
      ii = index - jj*NghostWidth - kk*NghostWidth*jjsize;

      // compute local
      global2local(ii,jj,kk,  i,j,k,idx,idy,idz, N);
      
      boundary_type = this->params.boundary_type_xmin;
      
      if(k >= kmin and k <= kmax and
	 j >= jmin and j <= jmax and
	 i >= 0    and i <ghostWidth) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            i0  = 2*ghostWidth-1-i;
            ii0 = N*i0 + N-1-idx;

            if (iVar==IU) sign=-ONE_F;
            // mirror DoFs idx <-> N-1-idx
            Udata(ii,jj,kk,iVar) = Udata(ii0,jj,kk,iVar)*sign;
            
          } else if( boundary_type == BC_NEUMANN ) {
            
            i0  = ghostWidth;
            ii0 = N*i0 + idx;

            Udata(ii,jj,kk,iVar) = Udata(ii0,jj,kk,iVar);
            
          } else { // periodic
            
            i0  = nx+i;
            ii0 = N*i0 + idx;

            Udata(ii,jj,kk,iVar) = Udata(ii0,jj,kk,iVar);
            
          }          
	  
        } // end for iVar
	
      } // end guard
    } // end FACE_XMIN

    if (faceId == FACE_XMAX) {
      
      // boundary xmax (index = i + j *ghostWidth + k * ghostWidth*jsize)
      // same i,j,k as xmin, except translation along x-axis
      kk = index / (NghostWidth*jjsize);
      jj = (index - kk*NghostWidth*jjsize) / NghostWidth;
      ii = index - jj*NghostWidth - kk*NghostWidth*jjsize;

      ii += (nx+ghostWidth)*N;
      
      // compute local
      global2local(ii,jj,kk,  i,j,k,idx,idy,idz, N);

      boundary_type = this->params.boundary_type_xmax;
      
      if(k >= kmin          and k <= kmax and
	 j >= jmin          and j <= jmax and
	 i >= nx+ghostWidth and i <= nx+2*ghostWidth-1) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            i0=2*nx+2*ghostWidth-1-i;
            ii0 = N*i0 +N-1-idx;

            if (iVar==IU) sign=-ONE_F;
            // mirror DoFs idx <-> N-1-idx
            Udata(ii,jj,kk,iVar) = Udata(ii0,jj,kk,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            i0  = nx+ghostWidth-1;
            ii0 = N*i0 + idx;

            Udata(ii,jj,kk,iVar) = Udata(ii0,jj,kk,iVar);
            
          } else { // periodic
            
            i0  = i-nx;
            ii0 = N*i0 + idx;

            Udata(ii,jj,kk,iVar) = Udata(ii0,jj,kk,iVar);
            
          }
          
        } // end for iVar

      } // end guard

    } // end FACE_XMAX

    if (faceId == FACE_YMIN) {

      // boundary ymin (index = i + j*isize + k*isize*ghostWidth)
      kk = index / (iisize*NghostWidth);
      jj = (index - kk*iisize*NghostWidth) / iisize;
      ii = index - jj*iisize - kk*iisize*NghostWidth;

      // compute local
      global2local(ii,jj,kk,  i,j,k,idx,idy,idz, N);

      boundary_type = this->params.boundary_type_ymin;
      
      if(k >= kmin and k <= kmax       and 
	 j >= 0    and j <  ghostWidth and
	 i >= imin and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ghostWidth-1-j;
            jj0 = N*j0 + N-1-idy;

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(ii,jj,kk,iVar) = Udata(ii,jj0,kk,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ghostWidth;
            jj0 = N*j0+idy;
            
            Udata(ii,jj,kk,iVar) = Udata(ii,jj0,kk,iVar);
            
          } else { // periodic
            
            j0  = ny+j;
            jj0 = N*j0+idy;
            
            Udata(ii,jj,kk,iVar) = Udata(ii,jj0,kk,iVar);
            
          }
          
        } // end for iVar
        
      } // end guard
      
    } // end FACE_YMIN
    
    if (faceId == FACE_YMAX) {
      
      // boundary ymax (index = i + j*isize + k*isize*ghostWidth)
      // same i,j,k as ymin, except translation along y-axis
      kk = index / (iisize*NghostWidth);
      jj = (index - kk*iisize*NghostWidth) / iisize;
      ii = index - jj*iisize - kk*iisize*NghostWidth;

      jj += (ny+ghostWidth)*N;

      // compute local
      global2local(ii,jj,kk,  i,j,k,idx,idy,idz, N);

      boundary_type = this->params.boundary_type_ymax;
      
      if(k >= kmin           and k <= kmax              and
	 j >= ny+ghostWidth  and j <= ny+2*ghostWidth-1 and
	 i >= imin           and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ny+2*ghostWidth-1-j;
            jj0 = N*j0 + N-1-idy;

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(ii,jj,kk,iVar) = Udata(ii,jj0,kk,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ny+ghostWidth-1;
            jj0 = N*j0 + idy;

            Udata(ii,jj,kk,iVar) = Udata(ii,jj0,kk,iVar);
            
          } else { // periodic
            
            j0  = j-ny;
            jj0 = N*j0 + idy;

            Udata(ii,jj,kk,iVar) = Udata(ii,jj0,kk,iVar);
            
          }
          
        } // end for iVar
		
      } // end guard

    } // end FACE_YMAX

    if (faceId == FACE_ZMIN) {
      
      // boundary zmin (index = i + j*isize + k*isize*jsize)
      kk = index / (iisize*jjsize);
      jj = (index - kk*iisize*jjsize) / iisize;
      ii = index - jj*iisize - kk*iisize*jjsize;

      // compute local
      global2local(ii,jj,kk,  i,j,k,idx,idy,idz, N);

      boundary_type = this->params.boundary_type_zmin;
      
      if(k >= 0    and k <  ghostWidth and
	 j >= jmin and j <= jmax       and
	 i >= imin and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            k0  = 2*ghostWidth-1-k;
            kk0 = N*k0 + N-1-idz;

            if (iVar==IW) sign=-ONE_F;
            // mirror DoFs idz <-> N-1-idz
            Udata(ii,jj,kk,iVar) = Udata(ii,jj,kk0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            k0  = ghostWidth;
            kk0 = N*k0 + idz;

            Udata(ii,jj,kk,iVar) = Udata(ii,jj,kk0,iVar);
            
          } else { // periodic
            
            k0  = nz+k;
            kk0 = N*k0 + idz;

            Udata(ii,jj,kk,iVar) = Udata(ii,jj,kk0,iVar);
            
          }
          
        } // end for iVar

      } // end guard

    } // end FACE_ZMIN
    
    if (faceId == FACE_ZMAX) {
      
      // boundary zmax (index = i + j*isize + k*isize*jsize)
      // same i,j,k as ymin, except translation along y-axis
      kk = index / (iisize*jjsize);
      jj = (index - kk*iisize*jjsize) / iisize;
      ii = index - jj*iisize - kk*iisize*jjsize;

      kk += (nz+ghostWidth)*N;

      // compute local
      global2local(ii,jj,kk,  i,j,k,idx,idy,idz, N);

      boundary_type = this->params.boundary_type_zmax;
      
      if(k >= nz+ghostWidth and k <= nz+2*ghostWidth-1 and
	 j >= jmin          and j <= jmax              and
	 i >= imin          and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            k0  = 2*nz+2*ghostWidth-1-k;
            kk0 = N*k0 + N-1-idz;
            
            if (iVar==IW) sign=-ONE_F;
            // mirror DoFs idz <-> N-1-idz
            Udata(ii,jj,kk,iVar) = Udata(ii,jj,kk0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            k0  = nz+ghostWidth-1;
            kk0 = N*k0 + idz;
            
            Udata(ii,jj,kk,iVar) = Udata(ii,jj,kk0,iVar);
            
          } else { // periodic
            
            k0  = k-nz;
            kk0 = N*k0 + idz;
            
            Udata(ii,jj,kk,iVar) = Udata(ii,jj,kk0,iVar);
            
          }
          
        } // end for iVar
	
      } // end guard

    } // end FACE_ZMAX

  } // end operator () - 3d
  
  DataArray Udata;
  
}; // MakeBoundariesFunctor_SDM

} // namespace sdm

#endif // SDM_BOUNDARIES_FUNCTORS_H_
