#include "IO_Writer.h"

#include <HydroParams.h>
#include <config/ConfigMap.h>

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
IO_Writer::IO_Writer(HydroParams& params,
		     ConfigMap& configMap,
		     int nbvar,
		     std::map<int, std::string>& variables_names) :
  IO_WriterBase(nbvar),
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
  hdf5_enabled = configMap.getBool("output","hdf5_enabled", true);
  
  // do we want Parallel NETCDF output ? Only valid/activated for MPI run
  pnetcdf_enabled = configMap.getBool("output","pnetcdf_enabled", false);
  
} // IO_Writer::IO_Writer


// =======================================================
// =======================================================
template<>
void IO_Writer::save_data_impl<DataArray2d>(DataArray2d             Udata,
					    DataArray2d::HostMirror Uhost,
					    int iStep)
{

  if (vtk_enabled)
    save_VTK_2D(Udata, Uhost, params, configMap, nbvar, variables_names, iStep);
  
} // IO_Writer::save_data_impl

// =======================================================
// =======================================================
template<>
void IO_Writer::save_data_impl<DataArray3d>(DataArray3d             Udata,
					    DataArray3d::HostMirror Uhost,
					    int iStep)
{

  if (vtk_enabled)
    save_VTK_3D(Udata, Uhost, params, configMap, nbvar, variables_names, iStep);
  
} // IO_Writer::save_data_impl

// =======================================================
// =======================================================
void IO_Writer::save_data(DataArray2d             Udata,
			  DataArray2d::HostMirror Uhost,
			  int iStep) {

  if (nbvar == 0)
    std::cerr << "nbvar is 0; no data dumped in file !\n";

  save_data_impl<DataArray2d>(Udata, Uhost, iStep);

} // IO_Writer::save_data
  
// =======================================================
// =======================================================
void IO_Writer::save_data(DataArray3d             Udata,
			  DataArray3d::HostMirror Uhost,
			  int iStep) {

  if (nbvar == 0)
    std::cerr << "nbvar is 0; no data dumped in file !\n";
  
  save_data_impl<DataArray3d>(Udata, Uhost, iStep);
  
} // IO_Writer::save_data

} // namespace io

} // namespace ppkMHD
