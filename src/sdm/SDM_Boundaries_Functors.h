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
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  MakeBoundariesFunctor_SDM(HydroParams           params,
			    SDM_Geometry<dim,N>   sdm_geom,
			    DataArray             Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    Udata(Udata),
    isize(params.isize),
    jsize(params.jsize),
    ksize(params.ksize)
  {};

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
    const int nbvar = this->params.nbvar;
    
    const int imin = this->params.imin;
    const int imax = this->params.imax;
    
    const int jmin = this->params.jmin;
    const int jmax = this->params.jmax;

    // compute iDof, iCell
    int iDof, iDof0, iCell, iCell0;
    index_to_iDof_iCell(index,N*N,iDof,iCell);

    // cell and Dof coordinates
    int i,j;
    int idx, idy;
    iDof_to_coord(iDof,N,idx,idy);

    int boundary_type;
    
    int i0, j0;
    int iVar;

    if (faceId == FACE_XMIN) {
      
      // boundary xmin
      boundary_type = this->params.boundary_type_xmin;

      // compute global
      j = iCell / ghostWidth;
      i = iCell - j*ghostWidth;

      if(j >= jmin and j <= jmax    and
	 i >= 0    and i <ghostWidth) {
        
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            i0  = 2*ghostWidth-1-i;
            iCell0 = i0 + isize*j;
            iDof0 = N-1-idx + N*idy;

            if (iVar==IU) sign=-ONE_F;
            
            // mirror DoFs idx <-> N-1-idx
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if( boundary_type == BC_NEUMANN ) {
            
            // TO BE MODIFIED: ghost cell DoFs must be extrapolated from
            // the inside
            i0  = ghostWidth;
            iCell0 = i0 + isize*j;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            i0  = nx+i;
            iCell0 = i0 + isize*j;
            iDof0 = iDof;

            //printf("XMIN %d %d - %d %d ---- %d %d | %d %d -- %d %d\n",i,j,idx,idy,iDof,iDof0,iCell,iCell0,i0,j);


            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }
	  
        } // end for iVar
	
      } // end guard
      
    } // end FACE_XMIN

    if (faceId == FACE_XMAX) {
      
      // boundary xmax
      boundary_type = this->params.boundary_type_xmax;

      // compute global
      j = iCell / ghostWidth;
      i = iCell - j*ghostWidth;
      i += (nx+ghostWidth);

      if(j >= jmin          and j <= jmax             and
	 i >= nx+ghostWidth and i <= nx+2*ghostWidth-1) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            i0  = 2*nx+2*ghostWidth-1-i;
            iCell0 = i0 + isize*j;
            iDof0 = N-1-idx + N*idy;
            
            if (iVar==IU) sign=-ONE_F;
            
            // mirror DoFs idx <-> N-1-idx
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            i0  = nx+ghostWidth-1;
            iCell0 = i0 + isize*j;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            i0  = i-nx;
            iCell0 = i0 + isize*j;
            iDof0 = iDof; 

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
          }
	  
        } // end for iVar

      } // end guard
      
    } // end FACE_XMAX
    
    if (faceId == FACE_YMIN) {
      
      // boundary ymin
      boundary_type = this->params.boundary_type_ymin;
      
      // compute global
      i = iCell / ghostWidth;
      j = iCell - i*ghostWidth;

      if(i >= imin and i <= imax    and
	 j >= 0    and j <ghostWidth) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ghostWidth-1-j;
            iCell0 = i + isize*j0;
            iDof0 = idx + N*(N-1-idy);

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ghostWidth;
            iCell0 = i + isize*j0;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            j0  = ny+j;
            iCell0 = i + isize*j0;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }
	  
        } // end for iVar

      } // end guard
      
    } // end FACE_YMIN

    if (faceId == FACE_YMAX) {

      // boundary ymax
      boundary_type = this->params.boundary_type_ymax;

      // compute global
      i = iCell / ghostWidth;
      j = iCell - i*ghostWidth;
      j += (ny+ghostWidth);

      if(i >= imin          and i <= imax              and
	 j >= ny+ghostWidth and j <= ny+2*ghostWidth-1) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ny+2*ghostWidth-1-j;
            iCell0 = i + isize*j0;
            iDof0 = idx + N*(N-1-idy);

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ny+ghostWidth-1;
            iCell0 = i + isize*j0;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            j0  = j-ny;
            iCell0 = i + isize*j0;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
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

    const int nbvar = this->params.nbvar;
    
    const int imin = this->params.imin;
    const int imax = this->params.imax;
    
    const int jmin = this->params.jmin;
    const int jmax = this->params.jmax;

    const int kmin = this->params.kmin;
    const int kmax = this->params.kmax;
    
    // compute iDof, iCell
    int iDof, iDof0, iCell, iCell0;
    index_to_iDof_iCell(index,N*N,iDof,iCell);

    // cell and Dof
    int i,j,k;
    int idx, idy, idz;
    iDof_to_coord(iDof,N,idx,idy,idz);
        
    int boundary_type;
    
    int i0, j0, k0;
    int iVar;
    
    if (faceId == FACE_XMIN) {
      
      // boundary xmin (index = i + j * ghostWidth + k * ghostWidth*jsize)
      k = iCell / (ghostWidth*jsize);
      j = (iCell - k*ghostWidth*jsize) / ghostWidth;
      i = iCell - j*ghostWidth - k*ghostWidth*jsize;

      boundary_type = this->params.boundary_type_xmin;
      
      if(k >= kmin and k <= kmax and
	 j >= jmin and j <= jmax and
	 i >= 0    and i <ghostWidth) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            i0  = 2*ghostWidth-1-i;
            iCell0 = i0 + isize*j + isize*jsize*k;
            iDof0 = N-1-idx + N*idy + N*N*idz;

            if (iVar==IU) sign=-ONE_F;
            // mirror DoFs idx <-> N-1-idx
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if( boundary_type == BC_NEUMANN ) {
            
            i0  = ghostWidth;
            iCell0 = i0 + isize*j + isize*jsize*k;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            i0  = nx+i;
            iCell0 = i0 + isize*j + isize*jsize*k;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }          
	  
        } // end for iVar
	
      } // end guard
    } // end FACE_XMIN

    if (faceId == FACE_XMAX) {
      
      // boundary xmax (index = i + j *ghostWidth + k * ghostWidth*jsize)
      // same i,j,k as xmin, except translation along x-axis
      k = iCell / (ghostWidth*jsize);
      j = (iCell - k*ghostWidth*jsize) / ghostWidth;
      i = iCell - j*ghostWidth - k*ghostWidth*jsize;

      i += (nx+ghostWidth);
      
      boundary_type = this->params.boundary_type_xmax;
      
      if(k >= kmin          and k <= kmax and
	 j >= jmin          and j <= jmax and
	 i >= nx+ghostWidth and i <= nx+2*ghostWidth-1) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            i0 = 2*nx+2*ghostWidth-1-i;
            iCell0 = i0 + isize*j + isize*jsize*k;
            iDof0 = N-1-idx + N*idy + N*N*idz;

            if (iVar==IU) sign=-ONE_F;
            // mirror DoFs idx <-> N-1-idx
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            i0  = nx+ghostWidth-1;
            iCell0 = i0 + isize*j + isize*jsize*k;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            i0  = i-nx;
            iCell0 = i0 + isize*j + isize*jsize*k;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }
          
        } // end for iVar

      } // end guard

    } // end FACE_XMAX

    if (faceId == FACE_YMIN) {

      // boundary ymin (index = i + j*isize + k*isize*ghostWidth)
      k = iCell / (isize*ghostWidth);
      j = (iCell - k*isize*ghostWidth) / isize;
      i = iCell - j*isize - k*isize*ghostWidth;

      boundary_type = this->params.boundary_type_ymin;
      
      if(k >= kmin and k <= kmax       and 
	 j >= 0    and j <  ghostWidth and
	 i >= imin and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ghostWidth-1-j;
            iCell0 = i + isize*j0 + isize*jsize*k;
            iDof0 = idx + N*(N-1-idy) + N*N*idz;

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ghostWidth;
            iCell0 = i + isize*j0 + isize*jsize*k;
            iDof0 = iDof;
            
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            j0  = ny+j;
            iCell0 = i + isize*j0 + isize*jsize*k;
            iDof0 = iDof;
            
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }
          
        } // end for iVar
        
      } // end guard
      
    } // end FACE_YMIN
    
    if (faceId == FACE_YMAX) {
      
      // boundary ymax (index = i + j*isize + k*isize*ghostWidth)
      // same i,j,k as ymin, except translation along y-axis
      k = iCell / (isize*ghostWidth);
      j = (iCell - k*isize*ghostWidth) / isize;
      i = iCell - j*isize - k*isize*ghostWidth;

      j += (ny+ghostWidth);

      boundary_type = this->params.boundary_type_ymax;
      
      if(k >= kmin           and k <= kmax              and
	 j >= ny+ghostWidth  and j <= ny+2*ghostWidth-1 and
	 i >= imin           and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            j0  = 2*ny+2*ghostWidth-1-j;
            iCell0 = i + isize*j0 + isize*jsize*k;
            iDof0 = idx + N*(N-1-idy) + N*N*idz;

            if (iVar==IV) sign=-ONE_F;
            // mirror DoFs idy <-> N-1-idy
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            j0  = ny+ghostWidth-1;
            iCell0 = i + isize*j0 + isize*jsize*k;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            j0  = j-ny;
            iCell0 = i + isize*j0 + isize*jsize*k;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }
          
        } // end for iVar
		
      } // end guard

    } // end FACE_YMAX

    if (faceId == FACE_ZMIN) {
      
      // boundary zmin (index = i + j*isize + k*isize*jsize)
      k = iCell / (isize*jsize);
      j = (iCell - k*isize*jsize) / isize;
      i = iCell - j*isize - k*isize*jsize;

      boundary_type = this->params.boundary_type_zmin;
      
      if(k >= 0    and k <  ghostWidth and
	 j >= jmin and j <= jmax       and
	 i >= imin and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
	  
          if ( boundary_type == BC_DIRICHLET ) {
            
            k0  = 2*ghostWidth-1-k;
            iCell0 = i + isize*j + isize*jsize*k0;
            iDof0 = idx + N*idy + N*N*(N-1-idz);

            if (iVar==IW) sign=-ONE_F;
            // mirror DoFs idz <-> N-1-idz
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            k0  = ghostWidth;
            iCell0 = i + isize*j + isize*jsize*k0;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            k0  = nz+k;
            iCell0 = i + isize*j + isize*jsize*k0;
            iDof0 = iDof;

            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }
          
        } // end for iVar

      } // end guard

    } // end FACE_ZMIN
    
    if (faceId == FACE_ZMAX) {
      
      // boundary zmax (index = i + j*isize + k*isize*jsize)
      // same i,j,k as ymin, except translation along y-axis
      k = iCell / (isize*jsize);
      j = (iCell - k*isize*jsize) / isize;
      i = iCell - j*isize - k*isize*jsize;

      k += (nz+ghostWidth);

      boundary_type = this->params.boundary_type_zmax;
      
      if(k >= nz+ghostWidth and k <= nz+2*ghostWidth-1 and
	 j >= jmin          and j <= jmax              and
	 i >= imin          and i <= imax) {
	
        for ( iVar=0; iVar<nbvar; iVar++ ) {
          real_t sign=1.0;
          
          if ( boundary_type == BC_DIRICHLET ) {
            
            k0  = 2*nz+2*ghostWidth-1-k;
            iCell0 = i + isize*j + isize*jsize*k0;
            iDof0 = idx + N*idy + N*N*(N-1-idz);
            
            if (iVar==IW) sign=-ONE_F;
            // mirror DoFs idz <-> N-1-idz
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar)*sign;
            
          } else if ( boundary_type == BC_NEUMANN ) {
            
            k0  = nz+ghostWidth-1;
            iCell0 = i + isize*j + isize*jsize*k0;
            iDof0 = iDof;
            
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          } else { // periodic
            
            k0  = k-nz;
            iCell0 = i + isize*j + isize*jsize*k0;
            iDof0 = iDof;
            
            Udata(iDof,iCell,iVar) = Udata(iDof0,iCell0,iVar);
            
          }
          
        } // end for iVar
	
      } // end guard

    } // end FACE_ZMAX

  } // end operator () - 3d
  
  DataArray Udata;
  const int isize, jsize, ksize;
  
}; // MakeBoundariesFunctor_SDM

} // namespace sdm

#endif // SDM_BOUNDARIES_FUNCTORS_H_
