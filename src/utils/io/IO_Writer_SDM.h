#ifndef IO_WRITER_SDM_H_
#define IO_WRITER_SDM_H_

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
//class HydroParams;
//class ConfigMap;
#include <shared/HydroParams.h>
#include <utils/config/ConfigMap.h>
#include <shared/utils.h>

#include "sdm/SDM_Geometry.h"

#include "IO_Writer.h"

#include "IO_VTK_SDM.h"


namespace ppkMHD { namespace io {

/**
 * Derived IO_Writer specific to Spectral Difference Method needs.
 */
template<int dim, int N>
class IO_Writer_SDM : public IO_Writer {

public:
  // alias to DataArray2d or DataArray3d
  using DataArray     = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;
  using DataArrayHost = typename DataArray::HostMirror;
  
  IO_Writer_SDM(HydroParams& params,
		ConfigMap& configMap,
		std::map<int, std::string>& variables_names,
		sdm::SDM_Geometry<dim,N> sdm_geom) :
    IO_Writer(params, configMap, variables_names),
    sdm_geom(sdm_geom) { };

  //! destructor
  virtual ~IO_Writer_SDM() {};

  //! Spectral Difference Method Geometry information
  sdm::SDM_Geometry<dim,N> sdm_geom;
  
  //! public interface to save data (override base class).
  void save_data_impl(DataArray     Udata,
		      DataArrayHost Uhost,
		      int iStep,
		      real_t time,
		      std::string debug_name)
  {
    
    if (vtk_enabled) {
      
#ifdef USE_MPI
      //save_VTK_2D_mpi(Udata, Uhost, params, configMap, params.nbvar, variables_names, iStep, debug_name);
#else
      save_VTK_SDM<N>(Udata, Uhost, params, configMap, sdm_geom, params.nbvar, variables_names, iStep, time);
#endif // USE_MPI
      
    }
    
    // #ifdef USE_HDF5
    //   if (hdf5_enabled) {
    
    // #ifdef USE_MPI
    //     ppkMHD::io::Save_HDF5_mpi<TWO_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    //     writer.save();
    // #else
    //     ppkMHD::io::Save_HDF5<TWO_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    //     writer.save();
    // #endif // USE_MPI
    
    //   }
    // #endif // USE_HDF5
    
    // #ifdef USE_PNETCDF
    //   if (pnetcdf_enabled) {
    //     ppkMHD::io::Save_PNETCDF<TWO_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    //     writer.save();    
    //   }
    // #endif // USE_PNETCDF
    
  } // IO_Writer_SDM::save_data_impl
  
}; // class IO_Writer_SDM


// =======================================================
// =======================================================
// template<int dim, int N>
// void IO_Writer::save_data_impl<dim,N>(DataArray2d             Udata,
// 				      DataArray2d::HostMirror Uhost,
// 				      int iStep,
// 				      real_t time,
// 				      std::string debug_name);

// =======================================================
// =======================================================
// template<int dim, int N>
// void IO_Writer::save_data_impl<dim,N>(DataArray3d             Udata,
// 				      DataArray3d::HostMirror Uhost,
// 				      int iStep,
// 				      real_t time,
// 				      std::string debug_name)
// {
  
//   if (vtk_enabled) {

// #ifdef USE_MPI
//     //save_VTK_3D_mpi(Udata, Uhost, params, configMap, params.nbvar, variables_names, iStep, debug_name);
// #else
//     //save_VTK_SDM(Udata, Uhost, params, configMap, sdm_geom, params.nbvar, variables_names, iStep, time);
// #endif // USE_MPI
    
//   }

// // #ifdef USE_HDF5
// //   if (hdf5_enabled) {

// // #ifdef USE_MPI
// //     ppkMHD::io::Save_HDF5_mpi<THREE_D> writer(Udata, Uhost, params, configMap, HYDRO_3D_NBVAR, variables_names, iStep, time, debug_name);
// //     writer.save();
// // #else
// //     ppkMHD::io::Save_HDF5<THREE_D> writer(Udata, Uhost, params, configMap, HYDRO_3D_NBVAR, variables_names, iStep, time, debug_name);
// //     writer.save();
// // #endif // USE_MPI
    
// //   }
// // #endif // USE_HDF5

// // #ifdef USE_PNETCDF
// //   if (pnetcdf_enabled) {
// //     ppkMHD::io::Save_PNETCDF<THREE_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
// //     writer.save();    
// //   }
// // #endif // USE_PNETCDF
  
// } // IO_Writer_SDM::save_data_impl

} // namespace io

} // namespace ppkMHD

#endif // IO_WRITER_SDM_H_
