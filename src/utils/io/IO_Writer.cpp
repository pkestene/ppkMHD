#include "IO_Writer.h"

#include <shared/HydroParams.h>
#include <utils/config/ConfigMap.h>
#include <shared/HydroState.h>

#include "IO_VTK.h"
#include "IO_HDF5.h"
//#include "IO_Pnetcdf.h"

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
IO_Writer::IO_Writer(HydroParams& params,
		     ConfigMap& configMap,
		     std::map<int, std::string>& variables_names) :
  IO_WriterBase(),
  params(params),
  configMap(configMap),
  variables_names(variables_names),
  vtk_enabled(true),
  hdf5_enabled(false),
  pnetcdf_enabled(false)
{
  
  // do we want VTK output ?
  vtk_enabled  = configMap.getBool("output","vtk_enabled", true);
  
  // do we want HDF5 output ?
  hdf5_enabled = configMap.getBool("output","hdf5_enabled", false);
  
  // do we want Parallel NETCDF output ? Only valid/activated for MPI run
  pnetcdf_enabled = configMap.getBool("output","pnetcdf_enabled", false);
  
} // IO_Writer::IO_Writer


// =======================================================
// =======================================================
template<>
void IO_Writer::save_data_impl<DataArray2d>(DataArray2d             Udata,
					    DataArray2d::HostMirror Uhost,
					    int iStep,
					    real_t time,
					    std::string debug_name)
{

  if (vtk_enabled) {
    
#ifdef USE_MPI
    save_VTK_2D_mpi(Udata, Uhost, params, configMap, params.nbvar, variables_names, iStep, debug_name);
#else
    save_VTK_2D(Udata, Uhost, params, configMap, params.nbvar, variables_names, iStep, debug_name);
#endif // USE_MPI

  }

  if (hdf5_enabled) {

    ppkMHD::io::Save_HDF5<TWO_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();

  }
  
} // IO_Writer::save_data_impl

// =======================================================
// =======================================================
template<>
void IO_Writer::save_data_impl<DataArray3d>(DataArray3d             Udata,
					    DataArray3d::HostMirror Uhost,
					    int iStep,
					    real_t time,
					    std::string debug_name)
{

  if (vtk_enabled) {

#ifdef USE_MPI
    save_VTK_3D_mpi(Udata, Uhost, params, configMap, params.nbvar, variables_names, iStep, debug_name);
#else
    save_VTK_3D(Udata, Uhost, params, configMap, params.nbvar, variables_names, iStep, debug_name);
#endif // USE_MPI
    
  }

  if (hdf5_enabled) {

    ppkMHD::io::Save_HDF5<THREE_D> writer(Udata, Uhost, params, configMap, HYDRO_3D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();

  }

  
} // IO_Writer::save_data_impl

// =======================================================
// =======================================================
void IO_Writer::save_data(DataArray2d             Udata,
			  DataArray2d::HostMirror Uhost,
			  int iStep,
			  real_t time,
			  std::string debug_name) {

  save_data_impl<DataArray2d>(Udata, Uhost, iStep, time, debug_name);

} // IO_Writer::save_data
  
// =======================================================
// =======================================================
void IO_Writer::save_data(DataArray3d             Udata,
			  DataArray3d::HostMirror Uhost,
			  int iStep,
			  real_t time,
			  std::string debug_name) {

  save_data_impl<DataArray3d>(Udata, Uhost, iStep, time, debug_name);
  
} // IO_Writer::save_data

} // namespace io

} // namespace ppkMHD
