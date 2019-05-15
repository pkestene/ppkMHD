#ifndef SDM_RUN_FUNCTORS_H_
#define SDM_RUN_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * A parallel functor to reset either a Solution point / Flux point
 * data arrary.
 */
template<int dim, int N>
class SDM_Erase_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  SDM_Erase_Functor(HydroParams         params,
		    SDM_Geometry<dim,N> sdm_geom,
		    DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    Udata(Udata)
  {
    iisize = Udata.extent(0);
    jjsize = Udata.extent(1);
    kksize = dim==3 ? Udata.extent(2) : 1;
  };

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
		    DataArray           Udata)
  {
    int64_t nbIter = Udata.extent(0) * Udata.extent(1);
    if (dim==3)
      nbIter *= Udata.extent(2);

    SDM_Erase_Functor functor(params, sdm_geom, Udata);
    Kokkos::parallel_for("SDM_Erase_Functor", nbIter, functor);
  }

  /*
   * 2D version.
   */
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {

    // global dofs index
    int ii,jj;
    index2coord(index,ii,jj,iisize,jjsize);
    
    Udata(ii,jj,ID) = 0.0;
    Udata(ii,jj,IP) = 0.0;
    Udata(ii,jj,IU) = 0.0;
    Udata(ii,jj,IV) = 0.0;
    
  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    // global index
    int ii,jj,kk;
    index2coord(index,ii,jj,kk,iisize,jjsize,kksize);

    Udata(ii,jj,kk,ID) = 0.0;
    Udata(ii,jj,kk,IP) = 0.0;
    Udata(ii,jj,kk,IU) = 0.0;
    Udata(ii,jj,kk,IV) = 0.0;
    Udata(ii,jj,kk,IW) = 0.0;
    
  } // end operator () - 3d
  
  DataArray Udata;
  int       iisize, jjsize, kksize;

}; // SDM_Erase_Functor

// =======================================================================
// =======================================================================
// /**
//  * Given the array (-dU/dt), perform update of Udata (conservative variable
//  * array).
//  *
//  * The minus sign comes from the conservative form ofthe Euler equation (all
//  * the terms are on the same side of the equation).
//  *
//  * \tparam dim dimension (2 or 3).
//  */
// template<int dim, int N>
// class SDM_Update_Functor : public SDMBaseFunctor<dim,N> {
  
// public:
//   using typename SDMBaseFunctor<dim,N>::DataArray;
//   using typename SDMBaseFunctor<dim,N>::HydroState;
  
//   SDM_Update_Functor(HydroParams         params,
// 		     SDM_Geometry<dim,N> sdm_geom,
// 		     DataArray           Udata,
// 		     DataArray           mdUdt,
// 		     real_t              dt) :
//     SDMBaseFunctor<dim,N>(params,sdm_geom),
//     Udata(Udata),
//     mdUdt(mdUdt),
//     dt(dt),
//     isize(params.isize),
//     jsize(params.jsize),
//     ksize(params.ksize),
//     ghostWidth(params.ghostWidth)
//   {};

//   // static method which does it all: create and execute functor
//   static void apply(HydroParams         params,
//                     SDM_Geometry<dim,N> sdm_geom,
//                     DataArray           Udata,
//                     DataArray           mdUdt,
//                     real_t              dt)
//   {
//     int64_t nbDofs = (dim==2) ? 
//       params.isize * params.jsize * N * N :
//       params.isize * params.jsize * params.ksize * N * N * N;
    
//     SDM_Update_Functor functor(params, sdm_geom,
//                                Udata, mdUdt, dt);
//     Kokkos::parallel_for("SDM_Update_Functor",nbDofs, functor);
//   }

//   //! functor for 2d 
//   template<int dim_ = dim>
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
//   {    
//     // global index
//     int ii,jj;
//     index2coord(index,ii,jj,isize*N,jsize*N);

//     // local cell index
//     int i,j;

//     // Dof index for flux
//     int idx,idy;

//     // mapping thread to solution Dof
//     global2local(ii,jj, i,j,idx,idy, N);

//     HydroState tmp;
    
//     if(j >= ghostWidth and j < jsize-ghostWidth  and
//        i >= ghostWidth and i < isize-ghostWidth ) {

//       Udata(ii,jj,ID) -= dt*mdUdt(ii,jj,ID);
//       Udata(ii,jj,IE) -= dt*mdUdt(ii,jj,IE);
//       Udata(ii,jj,IU) -= dt*mdUdt(ii,jj,IU);
//       Udata(ii,jj,IV) -= dt*mdUdt(ii,jj,IV);
      
//     } // end if guard
    
//   } // end operator ()
  
//   //! functor for 3d 
//   template<int dim_ = dim>
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index)  const
//   {
//     // global index
//     int ii,jj,kk;
//     index2coord(index,ii,jj,kk,isize*N,jsize*N,ksize*N);
    
//     // local cell index
//     int i,j,k;
    
//     // Dof index for flux
//     int idx,idy,idz;

//     // mapping thread to solution Dof
//     global2local(ii,jj,kk, i,j,k,idx,idy,idz, N);

//     HydroState tmp;

//     if(k >= ghostWidth and k < ksize-ghostWidth  and
//        j >= ghostWidth and j < jsize-ghostWidth  and
//        i >= ghostWidth and i < isize-ghostWidth ) {
	
//       Udata(ii,jj,kk,ID) -= dt*mdUdt(ii,jj,kk,ID);
//       Udata(ii,jj,kk,IE) -= dt*mdUdt(ii,jj,kk,IE);
//       Udata(ii,jj,kk,IU) -= dt*mdUdt(ii,jj,kk,IU);
//       Udata(ii,jj,kk,IV) -= dt*mdUdt(ii,jj,kk,IV);
//       Udata(ii,jj,kk,IW) -= dt*mdUdt(ii,jj,kk,IW);
      
//     } // end if guard
    
//   } // end operator ()
  
//   DataArray    Udata;
//   DataArray    mdUdt;
//   const real_t dt;
//   const int    isize, jsize, ksize;
//   const int    ghostWidth;

// }; // SDM_Update_Functor

// =======================================================================
// =======================================================================
/**
 * Perform an intermediate stage Runge-Kutta operation of the type
 * U_out = c0 * U_0 + c1 * U_1 + c2 * dt * U_2.
 *
 * \tparam dim dimension (2 or 3).
 * \tparam N SDM order
 */
template<int dim, int N>
class SDM_Update_RK_Functor : public SDMBaseFunctor<dim,N> {
  
public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;

  using coefs_t = Kokkos::Array<real_t,3>;
  
  SDM_Update_RK_Functor(HydroParams         params,
			SDM_Geometry<dim,N> sdm_geom,
			DataArray           Uout,
			DataArray           U_0,
			DataArray           U_1,
			DataArray           U_2,
			coefs_t             coefs,  
			real_t              dt) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    Uout(Uout),
    U_0(U_0),
    U_1(U_1),
    U_2(U_2),
    coefs(coefs),
    dt(dt),
    isize(params.isize),
    jsize(params.jsize),
    ksize(params.ksize),
    ghostWidth(params.ghostWidth)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    DataArray           Uout,
                    DataArray           U_0,
                    DataArray           U_1,
                    DataArray           U_2,
                    coefs_t             coefs,
                    real_t              dt)
  {
    int64_t nbDofs = (dim==2) ? 
      params.isize * params.jsize * N * N :
      params.isize * params.jsize * params.ksize * N * N * N;

    SDM_Update_RK_Functor functor(params, sdm_geom,
                                  Uout, U_0, U_1, U_2, coefs, dt);
    Kokkos::parallel_for("SDM_Update_RK_Functor",nbDofs, functor);
  }

    //! functor for 2d
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
  {
    const real_t c0   = coefs[0];
    const real_t c1   = coefs[1];
    const real_t c2dt = coefs[2]*dt;
    
    // global index
    int ii,jj;
    index2coord(index,ii,jj,isize*N,jsize*N);

    // local cell index
    int i,j;

    // Dof index for flux
    int idx,idy;

    // mapping thread to solution Dof
    global2local(ii,jj, i,j,idx,idy, N);

    HydroState tmp;
    
    if(j >= ghostWidth and j < jsize-ghostWidth  and
       i >= ghostWidth and i < isize-ghostWidth ) {

      tmp[ID] =
        c0   * U_0 (ii,jj,ID) +
        c1   * U_1 (ii,jj,ID) +
        c2dt * U_2 (ii,jj,ID) ;
      
      tmp[IE] =
        c0   * U_0 (ii,jj,IE) +
        c1   * U_1 (ii,jj,IE) +
        c2dt * U_2 (ii,jj,IE) ;
      
      tmp[IU] =
        c0   * U_0 (ii,jj,IU) +
        c1   * U_1 (ii,jj,IU) +
        c2dt * U_2 (ii,jj,IU) ;
      
      tmp[IV] =
        c0   * U_0 (ii,jj,IV) +
        c1   * U_1 (ii,jj,IV) +
        c2dt * U_2 (ii,jj,IV) ;
      
      Uout(ii,jj,ID) = tmp[ID];
      Uout(ii,jj,IE) = tmp[IE];
      Uout(ii,jj,IU) = tmp[IU];
      Uout(ii,jj,IV) = tmp[IV];
	  
    } // end if guard
    
  } // end operator ()
  
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index)  const
  {
    const real_t c0   = coefs[0];
    const real_t c1   = coefs[1];
    const real_t c2dt = coefs[2]*dt;

    // global index
    int ii,jj,kk;
    index2coord(index,ii,jj,kk,isize*N,jsize*N,ksize*N);
    
    // local cell index
    int i,j,k;
    
    // Dof index for flux
    int idx,idy,idz;

    // mapping thread to solution Dof
    global2local(ii,jj,kk, i,j,k,idx,idy,idz, N);

    HydroState tmp;

    if(k >= ghostWidth and k < ksize-ghostWidth  and
       j >= ghostWidth and j < jsize-ghostWidth  and
       i >= ghostWidth and i < isize-ghostWidth ) {
      
      tmp[ID] =
        c0   * U_0 (ii,jj,kk,ID) +
        c1   * U_1 (ii,jj,kk,ID) +
        c2dt * U_2 (ii,jj,kk,ID) ;
      
      tmp[IE] =
        c0   * U_0 (ii,jj,kk,IE) +
        c1   * U_1 (ii,jj,kk,IE) +
        c2dt * U_2 (ii,jj,kk,IE) ;
      
      tmp[IU] =
        c0   * U_0 (ii,jj,kk,IU) +
        c1   * U_1 (ii,jj,kk,IU) +
        c2dt * U_2 (ii,jj,kk,IU) ;
      
      tmp[IV] =
        c0   * U_0 (ii,jj,kk,IV) +
        c1   * U_1 (ii,jj,kk,IV) +
        c2dt * U_2 (ii,jj,kk,IV) ;
      
      tmp[IW] =
        c0   * U_0 (ii,jj,kk,IW) +
        c1   * U_1 (ii,jj,kk,IW) +
        c2dt * U_2 (ii,jj,kk,IW) ;
      
      Uout(ii,jj,kk,ID) = tmp[ID];
      Uout(ii,jj,kk,IE) = tmp[IE];
      Uout(ii,jj,kk,IU) = tmp[IU];
      Uout(ii,jj,kk,IV) = tmp[IV];
      Uout(ii,jj,kk,IW) = tmp[IW];
      
    } // end if guard
    
  } // end operator ()
  
  DataArray Uout;
  DataArray U_0;
  DataArray U_1;
  DataArray U_2;
  const coefs_t coefs;
  const real_t  dt;
  const int     isize, jsize, ksize;
  const int     ghostWidth;
  
}; // SDM_Update_RK_Functor

} // namespace sdm

#endif // SDM_RUN_FUNCTORS_H_

