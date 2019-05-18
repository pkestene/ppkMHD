#include "IO_VTK_SDM.h"

#include "shared/utils.h"  // for get_current_date
#include "shared/enums.h"  // for ComponentIndex3D (IX,IY,IZ)

namespace ppkMHD { namespace io {

// ==========================================================
// ==========================================================
void write_cells_data_2d(std::ostream& outFile,
                         sdm::DataArrayHost Uhost,
                         HydroParams& params,
                         ConfigMap& configMap,
                         const std::map<int, std::string>& variables_names,
                         uint64_t& offsetBytes,
                         int N)
{
  
  const int nx = params.nx;
  const int ny = params.ny;

  const int gw = params.ghostWidth;

  bool useDouble = sizeof(real_t) == sizeof(double) ? true : false;
  const char* dataType = useDouble ? "Float64" : "Float32";
  
  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  const int nbvar = variables_names.size();
  
  /*
   * write cell data.
   */
  outFile << "  <CellData>\n";

  // loop over scalar variables
  for ( int iVar=0; iVar<nbvar; iVar++ ) {
    
    outFile << "    <DataArray type=\"" << dataType
	    << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"" << ascii_or_binary << "\"";
    
    if (!outputVtkAscii) {
      outFile << " offset=\"" << offsetBytes << "\"";
    }

    outFile << " >\n";
    
    offsetBytes += sizeof(uint64_t) + sizeof(real_t)*nx*ny*N*N;
    
    if (outputVtkAscii) {
      // no ghost !!
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  // loop over sub-cells
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {
	      
	      real_t data = Uhost(idx+(gw+i)*N,
                                  idy+(gw+j)*N,
                                  iVar);
	      
	      outFile << data << " ";
	      
	    } // for idx
	  } // for idy
	  
	} // for i
      } // for j
      
      outFile << "\n";

    } // end outputVtkAscii
    
    outFile << "    </DataArray>\n";
    
  } // end for variables
  
  outFile << "  </CellData>\n";
  
} // write_cells_data_2d

// ==========================================================
// ==========================================================
void write_cells_data_3d(std::ostream& outFile,
                         sdm::DataArrayHost Uhost,
                         HydroParams& params,
                         ConfigMap& configMap,
                         const std::map<int, std::string>& variables_names,
                         uint64_t& offsetBytes,
                         int N)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const int gw = params.ghostWidth;

  bool useDouble = sizeof(real_t) == sizeof(double) ? true : false;
  const char* dataType = useDouble ? "Float64" : "Float32";
  
  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  const int nbvar = variables_names.size();

  /*
   * write cell data.
   */
  outFile << "  <CellData>\n";

  // loop over scalar variables
  for ( int iVar=0; iVar<nbvar; iVar++ ) {
    
    outFile << "    <DataArray type=\"" << dataType
	    << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"" << ascii_or_binary << "\"";

    if (!outputVtkAscii) {
      outFile << " offset=\"" << offsetBytes << "\"";
    }

    outFile<< " >\n";

    offsetBytes += sizeof(uint64_t) + sizeof(real_t)*nx*ny*nz*N*N*N;

    if (outputVtkAscii) {
      // no ghost !!
      for (int k=0; k<nz; ++k) {
	for (int j=0; j<ny; ++j) {
	  for (int i=0; i<nx; ++i) {
	    
	    // loop over sub-cells
	    for (int idz=0; idz<N; ++idz) {
	      for (int idy=0; idy<N; ++idy) {
		for (int idx=0; idx<N; ++idx) {

		  real_t data = Uhost(idx+(gw+i)*N,
                                      idy+(gw+j)*N, 
                                      idz+(gw+k)*N,
                                      iVar);
		  outFile << data << " ";
		  
		} // for idx
	      } // for idy
	    } // for idz
	    
	  } // for i
	} // for j
      } // for k
      
      outFile << "\n";
      
    } // end outputVtkAscii
    
    outFile << "    </DataArray>\n";
    
  } // end for variables
  
  outFile << "  </CellData>\n";
  
} // write_cells_data_3d

} // namespace io

} // namespace ppkMHD
