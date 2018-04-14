#include "IO_ReadWrite.h"

#include <shared/HydroParams.h>
#include <utils/config/ConfigMap.h>
#include <shared/HydroState.h>

#include "IO_VTK.h"

#ifdef USE_HDF5
#include "IO_HDF5.h"
#endif

#ifdef USE_PNETCDF
#include "IO_PNETCDF.h"
#endif // USE_PNETCDF

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
IO_ReadWrite::IO_ReadWrite(HydroParams& params,
			   ConfigMap& configMap,
			   std::map<int, std::string>& variables_names) :
  IO_ReadWriteBase(),
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
  
} // IO_ReadWrite::IO_ReadWrite

// =======================================================
// =======================================================
void IO_ReadWrite::save_data(DataArray2d             Udata,
			     DataArray2d::HostMirror Uhost,
			     int iStep,
			     real_t time,
			     std::string debug_name) {

  save_data_impl(Udata, Uhost, iStep, time, debug_name);

} // IO_ReadWrite::save_data
  
// =======================================================
// =======================================================
void IO_ReadWrite::save_data(DataArray3d             Udata,
			     DataArray3d::HostMirror Uhost,
			     int iStep,
			     real_t time,
			     std::string debug_name) {

  save_data_impl(Udata, Uhost, iStep, time, debug_name);
  
} // IO_ReadWrite::save_data

// =======================================================
// =======================================================
void IO_ReadWrite::save_data_impl(DataArray2d             Udata,
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

#ifdef USE_HDF5
  if (hdf5_enabled) {
    
#ifdef USE_MPI
    ppkMHD::io::Save_HDF5_mpi<TWO_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();
#else
    ppkMHD::io::Save_HDF5<TWO_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();
#endif // USE_MPI
    
  }
#endif // USE_HDF5

#ifdef USE_PNETCDF
  if (pnetcdf_enabled) {
    ppkMHD::io::Save_PNETCDF<TWO_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();    
  }
#endif // USE_PNETCDF
  
} // IO_ReadWrite::save_data_impl

// =======================================================
// =======================================================
void IO_ReadWrite::save_data_impl(DataArray3d             Udata,
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

#ifdef USE_HDF5
  if (hdf5_enabled) {

#ifdef USE_MPI
    ppkMHD::io::Save_HDF5_mpi<THREE_D> writer(Udata, Uhost, params, configMap, HYDRO_3D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();
#else
    ppkMHD::io::Save_HDF5<THREE_D> writer(Udata, Uhost, params, configMap, HYDRO_3D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();
#endif // USE_MPI
    
  }
#endif // USE_HDF5

#ifdef USE_PNETCDF
  if (pnetcdf_enabled) {
    ppkMHD::io::Save_PNETCDF<THREE_D> writer(Udata, Uhost, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    writer.save();    
  }
#endif // USE_PNETCDF
  
} // IO_ReadWrite::save_data_impl

// =======================================================
// =======================================================
void IO_ReadWrite::load_data(DataArray2d             Udata,
			     DataArray2d::HostMirror Uhost,
			     int& iStep,
			     real_t& time) {
  
  load_data_impl(Udata, Uhost, iStep, time);

} // IO_ReadWrite::save_data
  
// =======================================================
// =======================================================
void IO_ReadWrite::load_data(DataArray3d             Udata,
			     DataArray3d::HostMirror Uhost,
			     int& iStep,
			     real_t& time) {

  load_data_impl(Udata, Uhost, iStep, time);
  
} // IO_ReadWrite::load_data

// =======================================================
// =======================================================
void IO_ReadWrite::load_data_impl(DataArray2d             Udata,
				  DataArray2d::HostMirror Uhost,
				  int& iStep,
				  real_t& time)
{

  // TODO
  
} // IO_ReadWrite::load_data_impl - 2d

// =======================================================
// =======================================================
void IO_ReadWrite::load_data_impl(DataArray3d             Udata,
				  DataArray3d::HostMirror Uhost,
				  int& iStep,
				  real_t& time)
{
  
  // TODO
  
} // IO_ReadWrite::load_data_impl - 3d


} // namespace io

} // namespace ppkMHD