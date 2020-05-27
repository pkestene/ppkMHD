#ifndef IO_VTK_SDM_SHARED_H_
#define IO_VTK_SDM_SHARED_H_

#include <string>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

namespace ppkMHD { namespace io {

/**
 * Write VTK unstructured grid header.
 */
void write_vtu_header(std::ostream& outFile,	
		      ConfigMap& configMap);

/**
 * Write VTK unstructured grid metadata (date and time).
 */
void write_vtk_metadata(std::ostream& outFile,
			int iStep,
			real_t time);
/**
 * Write VTK unstructured grid footer.
 */
void write_vtu_footer(std::ostream& outFile);

void write_pvtu_header(std::string headerFilename,
		       std::string outputPrefix,
		       HydroParams& params,
		       ConfigMap& configMap,
		       int nbvar,
		       const std::map<int, std::string>& varNames,
		       int iStep,
		       bool is_flux_data_array = false);


} // namespace io

} // namespace ppkMHD

#endif // IO_VTK_SDM_SHARED_H_
