/**
 * Some specialized output routines to dump data in VTK unstructured grid
 * format for the High-order Spectral Difference Method schemes.
 */

#ifndef IO_VTK_SDM_H_
#define IO_VTK_SDM_H_

#include <map>
#include <string>

#include <shared/kokkos_shared.h>
#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

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

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid nodes locations (x,y,z).
 */
template<int N>
void write_nodes_location(std::ostream& outFile,
			  DataArray2d::HostMirror Uhost,
			  sdm::SDM_Geometry<2,N> sdm_geom,
			  HydroParams& params,
			  ConfigMap& configMap)
{
  
  const int nx = params.nx;
  const int ny = params.ny;

  const real_t xmin = params.xmin;
  const real_t ymin = params.ymin;
  const real_t dx = params.dx;
  const real_t dy = params.dy;
  
#ifdef USE_MPI
  const int i_mpi = params.myMpiPos[IX];
  const int j_mpi = params.myMpiPos[IY];
#else
  const int i_mpi = 0;
  const int j_mpi = 0;
#endif
  
  const int ghostWidth = params.ghostWidth;

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "binary";

  outFile << "  <Points>\n";
  outFile << "    <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\""
	  << ascii_or_binary << "\">\n";

  for (int j=0; j<ny; ++j) {
    for (int i=0; i<nx; ++i) {

      // cell offset
      real_t xo = xmin + (i+nx*i_mpi)*dx;
      real_t yo = ymin + (j+ny*j_mpi)*dy;
	
      for (int idy=0; idy<N+1; ++idy) {
	for (int idx=0; idx<N+1; ++idx) {

	  real_t x;
	  real_t y;

	  if (idx == 0) {
	    x = xo;
	  } else if ( idx == N) {
	    x = xo + dx;
	  } else { 
	  
	    x = xo + 0.5 * (sdm_geom.solution_pts_1d_host(idx-1) +
			    sdm_geom.solution_pts_1d_host(idx)   ) * dx;

	  }

	  if (idy == 0) {
	    y = yo;
	  } else if ( idy == N) {
	    y = yo + dy;
	  } else { 
	  
	    y = yo + 0.5 * (sdm_geom.solution_pts_1d_host(idy-1) +
			    sdm_geom.solution_pts_1d_host(idy  ) ) * dy;
	  }
	  
	  // now we can write node location
	  if (outputVtkAscii) {
	    outFile << x << " " << y << " " << 0.0 << "\n";
	  } else {
	    //outFile << ;
	  }
	}
      }
      
    } // end for i
  } // end for j
  
  outFile << "    </DataArray>\n";
  outFile << "  </Points>\n";
  
} // write_nodes_location - 2d

/**
 * Write VTK unstructured grid nodes connectivity + offsets + cell type (quad or hexahedron).
 */
template<int N>
void write_cells_connectivity(std::ostream& outFile,
			      DataArray2d::HostMirror Uhost,
			      sdm::SDM_Geometry<2,N> sdm_geom,
			      HydroParams& params,
			      ConfigMap& configMap)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const real_t xmin = params.xmin;
  const real_t ymin = params.ymin;
  const real_t dx = params.dx;
  const real_t dy = params.dy;
  
#ifdef USE_MPI
  const int i_mpi = params.myMpiPos[IX];
  const int j_mpi = params.myMpiPos[IY];
#else
  const int i_mpi = 0;
  const int j_mpi = 0;
#endif
  
  const int ghostWidth = params.ghostWidth;

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "binary";

  int nbNodesPerCell = (N+1)*(N+1); // in 2D
  int nbSubCells = N*N;
  
  // cell index
  int cell_index = 0;

  outFile << "  <Cells>\n";

  /*
   * CONNECTIVITY
   */
  outFile << "    <DataArray type=\"Int64\" Name=\"connectivity\" format=\"" << ascii_or_binary << "\""
	  << " >\n";

  for (int j=0; j<ny; ++j) {
    for (int i=0; i<nx; ++i) {

      int index = i+nx*j;

      // offset to the first nodes in this cell
      int offset = index * nbNodesPerCell;
      
      // loop over sub-cells
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {

	  outFile << offset+idx+  (N+1)* idy    << " "
		  << offset+idx+1+(N+1)* idy << " "
		  << offset+idx+1+(N+1)*(idy+1) << " "
		  << offset+idx  +(N+1)*(idy+1) << "\n";

	} // for idx
      } // for idy

      cell_index += nbSubCells;
      
    } // for i
  } // for j

  outFile << "    </DataArray>\n";

  /*
   * OFFSETS
   */
  outFile << "    <DataArray type=\"Int64\" Name=\"offsets\" format=\"" << ascii_or_binary << "\""
	  << " >\n";

  // number of nodes per cell is 4 in 2D
  for (int i=1; i<=nx*ny*N*N; ++i) {
    outFile << 4*i << " ";
  }
  outFile << "\n";
  
  outFile << "    </DataArray>\n";


  /*
   * CELL TYPES
   */
  outFile << "    <DataArray type=\"UInt8\" Name=\"types\" format=\"" << ascii_or_binary << "\""
	  << " >\n";

  // 9 means "Quad" - 12 means "Hexahedron"
  for (int i=0; i<nx*ny*N*N; ++i) {
    outFile << 9 << " ";
  }
  outFile << "\n";
  
  outFile << "    </DataArray>\n";

  /*
   * Close Cells section.
   */
  outFile << "  </Cells>\n";

} // write_cells_connectivity - 2d

/**
 * Write VTK unstructured grid cells data.
 */
template<int N>
void write_cells_data(std::ostream& outFile,
		      DataArray2d::HostMirror Uhost,
		      sdm::SDM_Geometry<2,N> sdm_geom,
		      HydroParams& params,
		      ConfigMap& configMap,
		      const std::map<int, std::string>& variables_names)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int gw = params.ghostWidth;

  bool useDouble = sizeof(real_t) == sizeof(double) ? true : false;
  const char* dataType = useDouble ? "Float64" : "Float32";
  
  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "binary";

  /*
   * write cell data.
   */
  outFile << "  <CellData>\n";

  // loop over scalar variables
  for ( int iVar=0; iVar<params.nbvar; iVar++) {
    
    outFile << "    <DataArray type=\"" << dataType
	    << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"" << ascii_or_binary << "\""
	    << " >\n";

    // no ghost !!
    for (int j=0; j<ny; ++j) {
      for (int i=0; i<nx; ++i) {
	
	// loop over sub-cells
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {
	    
	    outFile << Uhost(gw+i,gw+j, sdm::DofMap<2,N>(idx,idy, 0, iVar)) << " ";
	    
	  } // for idx
	} // for idy
	
      } // for i
    } // for j
    outFile << "\n";
    
    outFile << "    </DataArray>\n";
    
  } // end for variables
  
  outFile << "  </CellData>\n";
  
} // write_cells_data - 2d


/**
 * Write Parallel VTU header. 
 * Must be done by a single MPI process.
 *
 */
void write_pvtu_header(std::string headerFilename,
		       std::string outputPrefix,
		       HydroParams& params,
		       int nbvar,
		       const std::map<int, std::string>& varNames,
		       int iStep);

// ================================================================
// output routine (VTK file format, ASCII, VtkUnstructuredGrid)
// We use UnstructuredGrid here because, the mesh cells are unevenly
// split inti subcells, one per Dof of the SDM scheme.
//
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ================================================================
/**
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
// =======================================================
// =======================================================
template<int N>
void save_VTK_SDM(DataArray2d             Udata,
		  DataArray2d::HostMirror Uhost,
		  HydroParams& params,
		  ConfigMap& configMap,
		  sdm::SDM_Geometry<2,N> sdm_geom,
		  int nbvar,
		  const std::map<int, std::string>& variables_names,
		  int iStep,
		  real_t time)
{
  const int nx = params.nx;
  const int ny = params.ny;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;

  const int ghostWidth = params.ghostWidth;

  const int isize = params.isize;
  const int jsize = params.jsize;

  const int nbCells = isize*jsize;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;
  
  // concatenate file prefix + file number + suffix
  std::string filename;
  filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vtu";
  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header
  write_vtu_header(outFile, configMap);

  // write vtk metadata (time and iStep)
  write_vtk_metadata(outFile, iStep, time);
  
  // write mesh information
  // each "cell" actually has a N^2 mini-mesh
  int nbOfCells = nx*ny *  N   * N;
  int nbOfNodes = nx*ny * (N+1)*(N+1);
  
  outFile << "<Piece NumberOfPoints=\"" << nbOfNodes
	  <<"\" NumberOfCells=\"" << nbOfCells << "\" >\n";

  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "binary";

  /*
   * write nodes location
   */

  write_nodes_location<N>(outFile,Uhost,sdm_geom,params,configMap);

  write_cells_connectivity<N>(outFile, Uhost, sdm_geom, params, configMap);

  write_cells_data<N>(outFile, Uhost, sdm_geom, params, configMap, variables_names);
  
  outFile << " </Piece>\n";
  
  // if (outputVtkAscii) {

  //   outFile << "    <CellData>\n";

  //   // write data array (ascii), remove ghost cells
  //   for ( iVar=0; iVar<nbvar; iVar++) {
  //     outFile << "    <DataArray type=\"";
  //     if (useDouble)
  // 	outFile << "Float64";
  //     else
  // 	outFile << "Float32";
  //     outFile << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"ascii\" >\n";
      
  //     for (int index=0; index<nbCells; ++index) {
  // 	//index2coord(index,i,j,isize,jsize);
	
  // 	// enforce the use of left layout (Ok for CUDA)
  // 	// but for OpenMP, we will need to transpose
  // 	j = index / isize;
  // 	i = index - j*isize;
	
  // 	if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
  // 	    i>=imin+ghostWidth and i<=imax-ghostWidth) {
  // 	  outFile << Uhost(i, j, iVar) << " ";
  // 	}
  //     }
  //     outFile << "\n    </DataArray>\n";
  //   } // end for iVar

  //   outFile << "    </CellData>\n";
    
  //   // write footer
  //   outFile << "  </Piece>\n";
  //   outFile << "  </ImageData>\n";
  //   outFile << "</VTKFile>\n";

  // } else { // write data in binary format

  //   outFile << "    <CellData>" << std::endl;

  //   for (int iVar=0; iVar<nbvar; iVar++) {
  //     if (useDouble) {
  // 	outFile << "     <DataArray type=\"Float64\" Name=\"" ;
  //     } else {
  // 	outFile << "     <DataArray type=\"Float32\" Name=\"" ;
  //     }
  //     outFile << variables_names.at(iVar)
  // 	      << "\" format=\"appended\" offset=\""
  // 	      << iVar*nx*ny*sizeof(real_t)+iVar*sizeof(unsigned int)
  // 	      <<"\" />" << std::endl;
  //   }

  //   outFile << "    </CellData>" << std::endl;
  //   outFile << "  </Piece>" << std::endl;
  //   outFile << "  </ImageData>" << std::endl;
    
  //   outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

  //   // write the leading undescore
  //   outFile << "_";
  //   // then write heavy data (column major format)
  //   {
  //     unsigned int nbOfWords = nx*ny*sizeof(real_t);
  //     for (int iVar=0; iVar<nbvar; iVar++) {
  // 	outFile.write((char *)&nbOfWords,sizeof(unsigned int));
  // 	for (int j=jmin+ghostWidth; j<=jmax-ghostWidth; j++)
  // 	  for (int i=imin+ghostWidth; i<=imax-ghostWidth; i++) {
  // 	    real_t tmp = Uhost(i, j, iVar);
  // 	    outFile.write((char *)&tmp,sizeof(real_t));
  // 	  }
  //     }
  //   }

  //   outFile << "  </AppendedData>" << std::endl;
  //   outFile << "</VTKFile>" << std::endl;

  // } // end ascii/binary heavy data write

  write_vtu_footer(outFile);
  
  outFile.close();
  
} // end save_VTK<dim,N>


// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx+k*nx*ny)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void save_VTK_SDM_3D(DataArray3d             Udata,
		     DataArray3d::HostMirror Uhost,
		     HydroParams& params,
		     ConfigMap& configMap,
		     int nbvar,
		     const std::map<int, std::string>& variables_names,
		     int iStep,
		     std::string debug_name);


#ifdef USE_MPI
// /**
//  * \param[in] Udata device data to save
//  * \param[in,out] Uhost host data temporary array before saving to file
//  */
// void save_VTK_2D_mpi(DataArray2d             Udata,
// 		     DataArray2d::HostMirror Uhost,
// 		     HydroParams& params,
// 		     ConfigMap& configMap,
// 		     int nbvar,
// 		     const std::map<int, std::string>& variables_names,
// 		     int iStep,
// 		     std::string debug_name);

// /**
//  * \param[in] Udata device data to save
//  * \param[in,out] Uhost host data temporary array before saving to file
//  */
// void save_VTK_3D_mpi(DataArray3d             Udata,
// 		     DataArray3d::HostMirror Uhost,
// 		     HydroParams& params,
// 		     ConfigMap& configMap,
// 		     int nbvar,
// 		     const std::map<int, std::string>& variables_names,
// 		     int iStep,
// 		     std::string debug_name);

#endif // USE_MPI

} // namespace io

} // namespace ppkMHD

#endif // IO_VTK_SDM_H_
