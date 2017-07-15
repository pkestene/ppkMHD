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
 * (Perfrectly Matched Layer); see article
 * "Absorbing boundary conditions for the Euler and Navier–Stokes
 *  equations with the spectral difference method", by Ying Zhou, Z.J. Wang
 * Journal of Computational Physics, 229 (2010) 8733–8749
 */
template <int dim, int N, FaceIdType faceId>
class MakeBoundariesFunctor_SDM  : public SDMBaseFunctor<dim,N> {
  
public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  static constexpr auto dofMap = DofMap<dim,N>;
  
  MakeBoundariesFunctor_SDM(HydroParams           params,
			    SDM_Geometry<dim,N>   sdm_geom,
			    DataArray             Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    Udata(Udata)  {};
  
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
    const int nx = params.nx;
    const int ny = params.ny;
    
    const int ghostWidth = params.ghostWidth;
    const int nbvar = params.nbvar;
    
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
	
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {
	    
	    real_t sign=1.0;
	    for ( iVar=0; iVar<nbvar; iVar++ ) {
	      
	      if ( boundary_type == BC_DIRICHLET ) {
		i0=2*ghostWidth-1-i;
		if (iVar==IU) sign=-ONE_F;

		// mirror DoFs idx <-> N-1-idx
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i0  ,j  , dofMap(N-1-idx,idy,0,iVar))*sign;
		
	      } else if( boundary_type == BC_NEUMANN ) {

		// TO BE MODIFIED: ghost cell DoFs must be extrapolated from
		// the inside
		i0=ghostWidth;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i0  ,j  , dofMap(idx,idy,0,iVar));

	      } else { // periodic

		i0=nx+i;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i0  ,j  , dofMap(idx,idy,0,iVar));
		
	      }
	      
	  
	    } // end for iVar
	  } // end for idx
	} // end for idy
	
      } // end guard
      
    } // end FACE_XMIN

    if (faceId == FACE_XMAX) {
      
      // boundary xmax
      boundary_type = params.boundary_type_xmax;
      
      j = index / ghostWidth;
      i = index - j*ghostWidth;
      i += (nx+ghostWidth);

      if(j >= jmin          && j <= jmax             &&
	 i >= nx+ghostWidth && i <= nx+2*ghostWidth-1) {
	
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {

	    real_t sign=1.0;
	    for ( iVar=0; iVar<nbvar; iVar++ ) {
	      
	      if ( boundary_type == BC_DIRICHLET ) {

		i0=2*nx+2*ghostWidth-1-i;
		if (iVar==IU) sign=-ONE_F;

		// mirror DoFs idx <-> N-1-idx
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i0 ,j  , dofMap(N-1-idx,idy,0,iVar))*sign;
		
	      } else if ( boundary_type == BC_NEUMANN ) {

		i0=nx+ghostWidth-1;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i0 ,j  , dofMap(idx,idy,0,iVar));

	      } else { // periodic

		i0=i-nx;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i0 ,j  , dofMap(idx,idy,0,iVar));
	      }
	  
	    } // end for iVar
	  } // end for idx
	} // end for idy

      } // end guard
      
    } // end FACE_XMAX
    
    if (faceId == FACE_YMIN) {
      
      // boundary ymin
      boundary_type = params.boundary_type_ymin;
      
      i = index / ghostWidth;
      j = index - i*ghostWidth;

      if(i >= imin && i <= imax    &&
	 j >= 0    && j <ghostWidth) {
	
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {

	    real_t sign=1.0;
	    for ( iVar=0; iVar<nbvar; iVar++ ) {
	      if ( boundary_type == BC_DIRICHLET ) {

		j0=2*ghostWidth-1-j;
		if (iVar==IV) sign=-ONE_F;
		// mirror DoFs idy <-> N-1-idy
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i  ,j0 , dofMap(idx,N-1-idy,0,iVar))*sign;
		
	      } else if ( boundary_type == BC_NEUMANN ) {

		j0=ghostWidth;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i  ,j0 , dofMap(idx,idy,0,iVar));
		
	      } else { // periodic

		j0=ny+j;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i  ,j0 , dofMap(idx,idy,0,iVar));
		
	      }
	  
	    } // end for IVar
	  } // end for idx
	} // end for idy
	
      } // end guard
      
    } // end FACE_YMIN

    if (faceId == FACE_YMAX) {

      // boundary ymax
      boundary_type = params.boundary_type_ymax;
      
      i = index / ghostWidth;
      j = index - i*ghostWidth;
      j += (ny+ghostWidth);
      if(i >= imin          && i <= imax              &&
	 j >= ny+ghostWidth && j <= ny+2*ghostWidth-1) {
	
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {

	    real_t sign=1.0;
	    for ( iVar=0; iVar<nbvar; iVar++ ) {
	      
	      if ( boundary_type == BC_DIRICHLET ) {

		j0=2*ny+2*ghostWidth-1-j;
		if (iVar==IV) sign=-ONE_F;
		// mirror DoFs idy <-> N-1-idy
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i  ,j0  , dofMap(idx,N-1-idy,0,iVar))*sign;
		
	      } else if ( boundary_type == BC_NEUMANN ) {

		j0=ny+ghostWidth-1;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i  ,j0  , dofMap(idx,idy,0,iVar));
		
	      } else { // periodic

		j0=j-ny;
		Udata(i  ,j  , dofMap(idx,idy,0,iVar)) =
		  Udata(i  ,j0  , dofMap(idx,idy,0,iVar));
		
	      }
	      	      
	    } // end for iVar
	  } // end for idx
	} // end for idy

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

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;
    
    const int isize = params.isize;
    const int jsize = params.jsize;
    //const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    const int nbvar = params.nbvar;
    
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
    
    if (faceId == FACE_XMIN) {
      
      // boundary xmin (index = i + j * ghostWidth + k * ghostWidth*jsize)
      k = index / (ghostWidth*jsize);
      j = (index - k*ghostWidth*jsize) / ghostWidth;
      i = index - j*ghostWidth - k*ghostWidth*jsize;
      
      boundary_type = params.boundary_type_xmin;
      
      if(k >= kmin && k <= kmax &&
	 j >= jmin && j <= jmax &&
	 i >= 0    && i <ghostWidth) {
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {
	      
	      real_t sign=1.0;
	      for ( iVar=0; iVar<nbvar; iVar++ ) {
		
		if ( boundary_type == BC_DIRICHLET ) {

		  i0=2*ghostWidth-1-i;
		  if (iVar==IU) sign=-ONE_F;
		  // mirror DoFs idx <-> N-1-idx
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i0,j,k, dofMap(N-1-idx,idy,idz,iVar))*sign;

		} else if( boundary_type == BC_NEUMANN ) {

		  i0=ghostWidth;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i0,j,k, dofMap(idx,idy,idz,iVar));

		} else { // periodic

		  i0=nx+i;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i0,j,k, dofMap(idx,idy,idz,iVar));
		  
		}
		
	  
	      } // end for iVar
	    } // end for idx
	  } // end for idy
	} // end for idz
	
      } // end guard
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
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {

	      real_t sign=1.0;
	      for ( iVar=0; iVar<nbvar; iVar++ ) {
		
		if ( boundary_type == BC_DIRICHLET ) {

		  i0=2*nx+2*ghostWidth-1-i;
		  if (iVar==IU) sign=-ONE_F;
		  // mirror DoFs idx <-> N-1-idx
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i0,j,k, dofMap(N-1-idx,idy,idz,iVar))*sign;

		} else if ( boundary_type == BC_NEUMANN ) {

		  i0=nx+ghostWidth-1;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i0,j,k, dofMap(idx,idy,idz,iVar));

		} else { // periodic

		  i0=i-nx;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i0,j,k, dofMap(idx,idy,idz,iVar));

		}
				
	      } // end for iVar
	    } // end for idx
	  } // end for idy
	} // end for idz

      } // end guard
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
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {

	      real_t sign=1.0;
	      
	      for ( iVar=0; iVar<nbvar; iVar++ ) {
		if ( boundary_type == BC_DIRICHLET ) {

		  j0=2*ghostWidth-1-j;
		  if (iVar==IV) sign=-ONE_F;
		  // mirror DoFs idy <-> N-1-idy
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j0,k, dofMap(idx,N-1-idy,idz,iVar))*sign;

		} else if ( boundary_type == BC_NEUMANN ) {

		  j0=ghostWidth;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j0,k, dofMap(idx,idy,idz,iVar));

		} else { // periodic

		  j0=ny+j;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j0,k, dofMap(idx,idy,idz,iVar));

		}
				
	      } // end for iVar
	    } // end for idx
	  } // end for idy
	} // end for idz

      } // end guard
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
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {

	      real_t sign=1.0;
	      for ( iVar=0; iVar<nbvar; iVar++ ) {
		
		if ( boundary_type == BC_DIRICHLET ) {

		  j0=2*ny+2*ghostWidth-1-j;
		  if (iVar==IV) sign=-ONE_F;
		  // mirror DoFs idy <-> N-1-idy
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j0,k, dofMap(idx,N-1-idy,idz,iVar))*sign;

		} else if ( boundary_type == BC_NEUMANN ) {

		  j0=ny+ghostWidth-1;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j0,k, dofMap(idx,idy,idz,iVar));

		} else { // periodic

		  j0=j-ny;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j0,k, dofMap(idx,idy,idz,iVar));
		  
		}
				
	      } // end for iVar
	    } // end for idx
	  } // end for idy
	} // end for idz
		
      } // end guard
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
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {
	      
	      real_t sign=1.0;
	      
	      for ( iVar=0; iVar<nbvar; iVar++ ) {
		if ( boundary_type == BC_DIRICHLET ) {

		  k0=2*ghostWidth-1-k;
		  if (iVar==IW) sign=-ONE_F;
		  // mirror DoFs idz <-> N-1-idz
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j,k0, dofMap(idx,idy,N-1-idz,iVar))*sign;

		} else if ( boundary_type == BC_NEUMANN ) {

		  k0=ghostWidth;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j,k0, dofMap(idx,idy,idz,iVar));

		} else { // periodic

		  k0=nz+k;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j,k0, dofMap(idx,idy,idz,iVar));
		  
		}
				
	      } // end for iVar
	    } // end for idx
	  } // end for idy
	} // end for idz

      } // end guard
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
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {

	      real_t sign=1.0;
	
	      for ( iVar=0; iVar<nbvar; iVar++ ) {
		if ( boundary_type == BC_DIRICHLET ) {

		  k0=2*nz+2*ghostWidth-1-k;
		  if (iVar==IW) sign=-ONE_F;
		  // mirror DoFs idz <-> N-1-idz
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j,k0, dofMap(idx,idy,N-1-idz,iVar))*sign;

		} else if ( boundary_type == BC_NEUMANN ) {

		  k0=nz+ghostWidth-1;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j,k0, dofMap(idx,idy,idz,iVar));

		} else { // periodic

		  k0=k-nz;
		  Udata(i,j,k, dofMap(idx,idy,idz,iVar)) =
		    Udata(i,j,k0, dofMap(idx,idy,idz,iVar));

		}
				
	      } // end for iVar
	    } // end for idx
	  } // end for idy
	} // end for idz
	
      } // end guard
    } // end FACE_ZMAX

  } // end operator () - 3d
  
  HydroParams params;
  DataArray Udata;
  
}; // MakeBoundariesFunctor_SDM

} // namespace sdm

#endif // SDM_BOUNDARIES_FUNCTORS_H_
