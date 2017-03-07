#ifndef IO_VTK_H_
#define IO_VTK_H_

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
class HydroParams;
class ConfigMap;

namespace ppkMHD { namespace io {

// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void save_VTK_2D(DataArray2d             Udata,
		 DataArray2d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep);

// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx+k*nx*ny)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void save_VTK_3D(DataArray3d             Udata,
		 DataArray3d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep);

} // namespace io

} // namespace ppkMHD

#endif // IO_VTK_H_
