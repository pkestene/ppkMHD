#ifndef IO_HDF5_H_
#define IO_HDF5_H_

#include <iostream>

// for HDF5 file format output
#ifdef USE_HDF5
#include <hdf5.h>

#define HDF5_MESG(mesg)				\
  std::cerr << "HDF5 :" << mesg << std::endl;

#define HDF5_CHECK(val, mesg) do {				\
    if (!val) {							\
      std::cerr << "*** HDF5 ERROR ***\n";			\
      std::cerr << "    HDF5_CHECK (" << mesg << ") failed\n";	\
    }								\
  } while(0)

#endif // USE_HDF5

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
class HydroParams;
class ConfigMap;

namespace ppkMHD { namespace io {

/**
 * Write a wrapper file using the Xmdf file format (XML) to allow
 * Paraview/Visit to read these h5 files as a time series.
 *
 * \param[in] params a HydroParams struct (to retrieve geometry).
 * \param[in] totalNumberOfSteps The number of time steps computed.
 * \param[in] singleStep boolean; if true we only write header for
 *  the last step.
 * \param[in] ghostIncluded boolean; if true include ghost cells
 *
 * If library HDF5 is not available, do nothing.
 */
void writeXdmfForHdf5Wrapper(HydroParams& params,
			     ConfigMap& configMap,
			     bool mhdEnabled,
			     int totalNumberOfSteps,
			     bool singleStep,
			     bool ghostIncluded);

/**
 * Dump computation results (conservative variables) into a file
 * (HDF5 file format) file extension is h5. File can be viewed by
 * hdfview; see also h5dump.
 *
 * \sa writeXdmfForHdf5Wrapper this routine write a Xdmf wrapper file for paraview.
 *
 * If library HDF5 is not available, do nothing.
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
void save_HDF5_2D(DataArray2d             Udata,
		  DataArray2d::HostMirror Uhost,
		  HydroParams& params,
		  ConfigMap& configMap,
		  bool mhdEnabled,
		  int nbvar,
		  const std::map<int, std::string>& variables_names,
		  int iStep,
		  real_t totalTime,
		  std::string debug_name);


} // namespace io

} // namespace ppkMHD

#endif // IO_HDF5_H_
