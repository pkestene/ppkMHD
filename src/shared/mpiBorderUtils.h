/**
 * \file mpiBorderUtils.h
 * \brief Some utility routines dealing with MPI border buffers.
 *
 * \date 13 Oct 2010
 * \author Pierre Kestener
 *
 */
#ifndef MPI_BORDER_UTILS_H_
#define MPI_BORDER_UTILS_H_

#include "shared/kokkos_shared.h"
#include "shared/enums.h"

namespace ppkMHD {

/**
 * \class CopyBorderBuf_To_DataArray
 *
 * Copy a border buffer (as received by MPI communications) into the
 * right location (given by template parameter boundaryLoc).
 * Here we assume U is a DataArray. 
 *
 * template parameters:
 * @tparam boundaryLoc : destination boundary location 
 *                       used to check array dimensions and set offset
 * @tparam dimType     : triggers 2D or 3D specific treatment
 *
 * argument parameters:
 * @param[out] U reference to a hydro simulations array (destination array)
 * @param[in]  b reference to a border buffer (source array)
 * @param[in]  ghostWidth is the number of ghost cells
 *
 */
template<
  BoundaryLocation boundaryLoc,
  DimensionType    dimType>
class CopyBorderBuf_To_DataArray {
  
public:
  //! Decide at compile-time which data array to use
  using DataArray  = typename std::conditional<dimType==TWO_D,DataArray2d,DataArray3d>::type;

  
  CopyBorderBuf_To_DataArray(DataArray U,
			     DataArray b,
			     int       ghostWidth) :
    U(U), b(b), ghostWidth(ghostWidth) {};
  
  
  template<DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dimType_==TWO_D, int>::type&  index) const
  {

    const int isize = U.dimension_0();
    const int jsize = U.dimension_1();
    int i,j;
    index2coord(index,i,j,isize,jsize);


    /*
     * Proceed with copy.
     */
    int offset = 0;
    if (boundaryLoc == XMAX)
      offset = U.dimension_0()-ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.dimension_1()-ghostWidth;
    
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      
      for (int nVar=0; nVar<U.nvar(); ++nVar)
	for (int offset_i=0; offset_i<ghostWidth; ++offset_i) {
	  U(offset+offset_i  ,j,nVar) = b(offset_i,j,nVar);
	}
      
    } else if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
      
      for (int nVar=0; nVar<U.nvar(); ++nVar)
	for (int offset_j=0; offset_j<ghostWidth; ++offset_j) {
	  U(i,offset+offset_j  ,nVar) = b(i,offset_j,nVar);
	}
      
    }
    
  } // operator() - 2D
  
  template<DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dimType_==THREE_D, int>::type&  index) const
  {

    const int isize = U.dimension_0();
    const int jsize = U.dimension_1();
    const int ksize = U.dimension_2();
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);
    
    /*
     * Proceed with copy.
     */
    int offset = 0;
    if (boundaryLoc == XMAX)
      offset = U.dimension_0()-ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.dimension_1()-ghostWidth;
    if (boundaryLoc == ZMAX)
      offset = U.dimension_2()-ghostWidth;
    
    
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      	
      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_i=0; offset_i<ghostWidth; ++offset_i) {
	  U(offset+offset_i  ,j,k,nVar) = b(offset_i,j,k,nVar);
	}
      }
      
    } else if (boundaryLoc == YMIN or boundaryLoc == YMAX) {
      
      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_j=0; offset_j<ghostWidth; ++offset_j) {
	  U(i,offset+offset_j  ,k,nVar) = b(i,offset_j,k,nVar);
	}
      }
      
    } else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {
      
      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_k=0; offset_k<ghostWidth; ++offset_k) {
	  U(i,j,offset+offset_k  ,nVar) = b(i,j,offset_k,nVar);
	}
      }
      
    }
    
  } // operator() - 3D

  DataArray U;
  DataArray b;
  int       ghostWidth;
  
}; // class CopyBorderBuf_To_DataArray

/**
 * \class CopyDataArray_To_BorderBuf
 * 
 * Copy array border to a border buffer (to be sent by MPI communications) 
 * Here we assume U is a <b>DataArray</b>. 
 * \sa copyBorderBufSendToHostArray
 *
 * template parameters:
 * @tparam boundaryLoc : boundary location in source Array
 * @tparam dimType     : triggers 2D or 3D specific treatment
 *
 * argument parameters:
 * @param[out] b reference to a border buffer (destination array)
 * @param[in]  U reference to a hydro simulations array (source array)
 * @param[in]  ghostWidth is the number of ghost cells
 */
template<
  BoundaryLocation boundaryLoc,
  DimensionType    dimType>
class CopyDataArray_To_BorderBuf {

public:
  //! Decide at compile-time which data array to use
  using DataArray  = typename std::conditional<dimType==TWO_D,DataArray2d,DataArray3d>::type;
  
  CopyDataArray_To_BorderBuf(DataArray b,
			     DataArray U,
			     int       ghostWidth) :
    b(b), U(U), ghostWidth(ghostWidth) {};
    
  template<DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dimType_==TWO_D, int>::type&  index) const
  {
    
    const int isize = U.dimension_0();
    const int jsize = U.dimension_1();
    int i,j;
    index2coord(index,i,j,isize,jsize);

    /*
     * Proceed with copy
     */
    int offset = ghostWidth;
    if (boundaryLoc == XMAX)
      offset = U.dimension_0()-2*ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.dimension_1()-2*ghostWidth;

    /*
     * simple copy when PERIODIC or COPY
     */      
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      
      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_i=0; offset_i<ghostWidth; ++offset_i) {
	  b(offset_i,j,nVar) = U(offset+offset_i  ,j,nVar);
	}
      }
      
    } else if (boundaryLoc == YMIN or boundaryLoc == YMAX) {

      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_j=0; offset_j<ghostWidth; ++offset_j) {
	  b(i,offset_j,nVar) = U(i,offset+offset_j  ,nVar);
	}
      }

    }
      
  } // operator() - 2D

  template<DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dimType_==THREE_D, int>::type&  index) const
  {

    const int isize = U.dimension_0();
    const int jsize = U.dimension_1();
    const int ksize = U.dimension_2();
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    /*
     * Proceed with copy
     */
    int offset = ghostWidth;
    if (boundaryLoc == XMAX)
      offset = U.dimension_0()-2*ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.dimension_1()-2*ghostWidth;
    if (boundaryLoc == ZMAX)
      offset = U.dimension_2()-2*ghostWidth;
    

    /*
     * simple copy when PERIODIC or COPY
     */      
    if (boundaryLoc == XMIN or boundaryLoc == XMAX) {
      
      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_i=0; offset_i<ghostWidth; ++offset_i) {
	  b(offset_i,j,k,nVar) = U(offset+offset_i  ,j,k,nVar);
	}
      }
	
    } else if (boundaryLoc == YMIN or boundaryLoc == YMAX) {

      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_j=0; offset_j<ghostWidth; ++offset_j) {
	  b(i,offset_j,k,nVar) = U(i,offset+offset_j  ,k,nVar);
	}
      }
      
    } else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX) {

      for (int nVar=0; nVar<U.nvar(); ++nVar) {
	for (int offset_k=0; offset_k<ghostWidth; ++offset_k) {
	  b(i,j,offset_k,nVar) = U(i,j,offset+offset_k  ,nVar);
	}
      }
      
    } // end (boundaryLoc == ZMIN or boundaryLoc == ZMAX)
      
  } // operator() - 3D

  DataArray b;
  DataArray U;
  int       ghostWidth;

}; // class CopyDataArray_To_BorderBuf

} // namespace ppkMHD

#endif // MPI_BORDER_UTILS_H_