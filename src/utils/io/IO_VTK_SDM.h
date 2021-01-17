/**
 * Some specialized output routines to dump data in VTK unstructured grid
 * format for the High-order Spectral Difference Method schemes.
 */

#ifndef IO_VTK_SDM_H_
#define IO_VTK_SDM_H_

#include <map>
#include <string>

#include <cstdint>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include <iostream>
#include <fstream>

#include "utils/io/IO_VTK_SDM_shared.h"

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid nodes locations (x,y,z).
 *
 * \param[in,out] offsetBytes is incremented by the size of data written (only
 *                useful for appended binary data).
 */
template<int N>
void write_nodes_location(std::ostream& outFile,
			  sdm::SDM_Geometry<2,N> sdm_geom,
			  HydroParams& params,
			  ConfigMap& configMap,
			  uint64_t& offsetBytes)
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
  
  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  //bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);
  
  outFile << "  <Points>\n";
  outFile << "    <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\""
	  << ascii_or_binary << "\"";

  if (!outputVtkAscii) {
    outFile << " offset=\"" << 0 << "\"";
  }
  
  outFile << ">" << "\n";

  if (outputVtkAscii) {
    for (int j=0; j<ny; ++j) {
      for (int i=0; i<nx; ++i) {
	
	// cell offset
	real_t xo = xmin + (i+nx*i_mpi)*dx;
	real_t yo = ymin + (j+ny*j_mpi)*dy;
	
	for (int idy=0; idy<N+1; ++idy) {
	  for (int idx=0; idx<N+1; ++idx) {
	    
	    float x, y;
	    
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
	    
	    // now we can write node locations
	    outFile << x << " " << y << " " << 0.0 << "\n";
	  } // end for idx
	} // end for idy
	
      } // end for i
    } // end for j
    
  } // outputVtkAscii
    
  outFile << "    </DataArray>\n";
  outFile << "  </Points>\n";

  offsetBytes += sizeof(uint64_t) + sizeof(float)*nx*ny*(N+1)*(N+1)*3;
  
} // write_nodes_location - 2d

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid nodes locations (x,y,z) - 3D.
 *
 * \param[in,out] offsetBytes is incremented by the size of data written (only
 *                useful for appended binary data).
 */
template<int N>
void write_nodes_location(std::ostream& outFile,
			  sdm::SDM_Geometry<3,N> sdm_geom,
			  HydroParams& params,
			  ConfigMap& configMap,
			  uint64_t& offsetBytes)
{
  
  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const real_t xmin = params.xmin;
  const real_t ymin = params.ymin;
  const real_t zmin = params.zmin;

  const real_t dx = params.dx;
  const real_t dy = params.dy;
  const real_t dz = params.dz;
  
#ifdef USE_MPI
  const int i_mpi = params.myMpiPos[IX];
  const int j_mpi = params.myMpiPos[IY];
  const int k_mpi = params.myMpiPos[IZ];
#else
  const int i_mpi = 0;
  const int j_mpi = 0;
  const int k_mpi = 0;
#endif
  
  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  outFile << "  <Points>\n";
  outFile << "    <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\""
	  << ascii_or_binary << "\"";

  if (!outputVtkAscii) {
    outFile << " offset=\"" << 0 << "\"";
  }
  
  outFile << ">" << "\n";

  if (outputVtkAscii) {
    for (int k=0; k<nz; ++k) {
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  // cell offset
	  real_t xo = xmin + (i+nx*i_mpi)*dx;
	  real_t yo = ymin + (j+ny*j_mpi)*dy;
	  real_t zo = zmin + (k+nz*k_mpi)*dz;
	  
	  for (int idz=0; idz<N+1; ++idz) {
	    for (int idy=0; idy<N+1; ++idy) {
	      for (int idx=0; idx<N+1; ++idx) {
		
		real_t x,y,z;
		
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
		
		if (idz == 0) {
		  z = zo;
		} else if ( idz == N) {
		  z = zo + dz;
		} else { 
		  
		  z = zo + 0.5 * (sdm_geom.solution_pts_1d_host(idz-1) +
				  sdm_geom.solution_pts_1d_host(idz  ) ) * dz;
		}
		
		// now we can write node location
		outFile << x << " " << y << " " << z << "\n";
		
	      } // for idx
	    } // for idy
	  } // for idz
	  
	} // end for i
      } // end for j
    } // end for k
    
  } // end outputVtkAscii
  
  outFile << "    </DataArray>\n";
  outFile << "  </Points>\n";
  
  offsetBytes += sizeof(uint64_t) + sizeof(float)*nx*ny*nz*(N+1)*(N+1)*(N+1)*3;

} // write_nodes_location - 3d

/**
 * Write VTK unstructured grid nodes connectivity + offsets + cell type
 * (quad in 2D).
 *
 * returned value is only useful when using appended binary data, the returned
 * is actually the currently value of "offset".
 *
 */
template<int N>
void write_cells_connectivity(std::ostream& outFile,
			      sdm::SDM_Geometry<2,N> sdm_geom,
			      HydroParams& params,
			      ConfigMap& configMap,
			      uint64_t& offsetBytes)
{

  const int nx = params.nx;
  const int ny = params.ny;

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  int nbNodesPerCell = (N+1)*(N+1); // in 2D
  int nbSubCells = N*N;
  
  outFile << "  <Cells>\n";

  /*
   * CONNECTIVITY
   */
  outFile << "    <DataArray type=\"Int64\" Name=\"connectivity\" format=\"" << ascii_or_binary << "\"";
  
  if (!outputVtkAscii) {
    outFile << " offset=\"" << offsetBytes << "\"";
  }

  outFile << " >\n";

  offsetBytes += sizeof(uint64_t) + sizeof(uint64_t)*nx*ny*nbSubCells*4;

  if (outputVtkAscii) {

    // cell index
    int cell_index = 0;
    
    for (int j=0; j<ny; ++j) {
      for (int i=0; i<nx; ++i) {
	
	uint64_t index = i+nx*j;
	
	// offset to the first nodes in this cell
	uint64_t offset = index * nbNodesPerCell;
	
	// loop over sub-cells
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {
	    
	    uint64_t i0,i1,i2,i3;
	    i0 = offset+idx+  (N+1)* idy;
	    i1 = offset+idx+1+(N+1)* idy;
	    i2 = offset+idx+1+(N+1)*(idy+1);
	    i3 = offset+idx  +(N+1)*(idy+1);

	    outFile << i0 << " " << i1 << " " << i2 << " " << i3 << "\n";
	    	    
	  } // for idx
	} // for idy
	
	cell_index += nbSubCells;
	
      } // for i
    } // for j
  } // end outputVtkAscii
  
  outFile << "    </DataArray>\n";

  /*
   * OFFSETS
   */
  outFile << "    <DataArray type=\"Int64\" Name=\"offsets\" format=\"" << ascii_or_binary << "\"";

  if (!outputVtkAscii) {
    outFile << " offset=\"" << offsetBytes << "\"";
  }
  
  outFile << " >\n";

  offsetBytes += sizeof(uint64_t) + sizeof(uint64_t)*nx*ny*N*N;

  if (outputVtkAscii) {
    // number of nodes per cell is 4 in 2D
    for (int i=1; i<=nx*ny*N*N; ++i) {
      uint64_t cell_offset = 4*i;
      outFile << cell_offset << " ";
    }
    outFile << "\n";
  }
  
  outFile << "    </DataArray>\n";


  /*
   * CELL TYPES
   */
  outFile << "    <DataArray type=\"UInt8\" Name=\"types\" format=\"" << ascii_or_binary << "\"";

  if (!outputVtkAscii) {
    outFile << " offset=\"" << offsetBytes << "\"";
  }
  
  outFile << " >\n";

  offsetBytes += sizeof(uint64_t) + sizeof(unsigned char)*nx*ny*N*N;
  
  if (outputVtkAscii) {
    // 9 means "Quad" - 12 means "Hexahedron"
    for (int i=0; i<nx*ny*N*N; ++i) {
      outFile << 9 << " ";
    }

    outFile << "\n";
  }
  
  outFile << "    </DataArray>\n";

  /*
   * Close Cells section.
   */
  outFile << "  </Cells>\n";

} // write_cells_connectivity - 2d

/**
 * Write VTK unstructured grid nodes connectivity + offsets + cell type
 * (hexahedron in 3D).
 *
 * returned value is only useful when using appended binary data, the returned
 * is actually the currently value of "offset".
 *
 */
template<int N>
void write_cells_connectivity(std::ostream& outFile,
			      sdm::SDM_Geometry<3,N> sdm_geom,
			      HydroParams& params,
			      ConfigMap& configMap,
			      uint64_t& offsetBytes)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  int nbNodesPerCell = (N+1)*(N+1)*(N+1); // in 3D
  int nbSubCells = N*N*N;

  int N1=N+1;
  int N2=(N+1)*(N+1);
    
  outFile << "  <Cells>\n";

  /*
   * CONNECTIVITY
   */
  outFile << "    <DataArray type=\"Int64\" Name=\"connectivity\" format=\"" << ascii_or_binary << "\"";

  if (!outputVtkAscii) {
    outFile << " offset=\"" << offsetBytes << "\"";
  }

  outFile << " >\n";

  offsetBytes += sizeof(uint64_t) + sizeof(uint64_t)*nx*ny*nz*nbSubCells*8;

  if (outputVtkAscii) {

    // cell index
    int cell_index = 0;

    for (int k=0; k<nz; ++k) {
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  int index = i+nx*j+nx*ny*k;
	  
	  // offset to the first nodes in this cell
	  int offset = index * nbNodesPerCell;
	  
	  // loop over sub-cells
	  for (int idz=0; idz<N; ++idz) {
	    for (int idy=0; idy<N; ++idy) {
	      for (int idx=0; idx<N; ++idx) {
		
		uint64_t i0,i1,i2,i3,i4,i5,i6,i7;
		i0 = offset+idx  +N1* idy   + N2* idz   ;
		i1 = offset+idx+1+N1* idy   + N2* idz   ;
		i2 = offset+idx+1+N1*(idy+1)+ N2* idz   ;
		i3 = offset+idx  +N1*(idy+1)+ N2* idz   ;
		i4 = offset+idx  +N1* idy   + N2*(idz+1);
		i5 = offset+idx+1+N1* idy   + N2*(idz+1);
		i6 = offset+idx+1+N1*(idy+1)+ N2*(idz+1);
		i7 = offset+idx  +N1*(idy+1)+ N2*(idz+1);

		outFile << i0 << " " << i1 << " " << i2 << " " << i3 << " "
			<< i4 << " " << i5 << " " << i6 << " " << i7 << "\n";
		
	      } // for idx
	    } // for idy
	  } // for idz
	  
	  cell_index += nbSubCells;
	  
	} // for i
      } // for j
    } // for k
    
  } // end outputVtkAscii
  
  outFile << "    </DataArray>\n";

  /*
   * OFFSETS
   */
  outFile << "    <DataArray type=\"Int64\" Name=\"offsets\" format=\"" << ascii_or_binary << "\"";

  if (!outputVtkAscii) {
    outFile << " offset=\"" << offsetBytes << "\"";
  }

  outFile << " >\n";

  offsetBytes += sizeof(uint64_t) + sizeof(uint64_t)*nx*ny*nz*nbSubCells;

  if (outputVtkAscii) {
    // number of nodes per cell is 8 in 3D
    for (int i=1; i<=nx*ny*nz*nbSubCells; ++i) {
      uint64_t cell_offset = 8*i;
      outFile << cell_offset << " ";
    }
    outFile << "\n";
  }
  
  outFile << "    </DataArray>\n";


  /*
   * CELL TYPES
   */
  outFile << "    <DataArray type=\"UInt8\" Name=\"types\" format=\"" << ascii_or_binary << "\"";

  if (!outputVtkAscii) {
    // before offset is cells location + connectivity + offsets
    outFile << " offset=\"" << offsetBytes << "\"";
  }

  outFile << " >\n";

  offsetBytes += sizeof(uint64_t) + sizeof(unsigned char)*nx*ny*nz*nbSubCells;

  if (outputVtkAscii) {
    // 9 means "Quad" - 12 means "Hexahedron"
    for (int i=0; i<nx*ny*nz*nbSubCells; ++i) {
      outFile << 12 << " ";
    }
    outFile << "\n";
  }
  
  outFile << "    </DataArray>\n";

  /*
   * Close Cells section.
   */
  outFile << "  </Cells>\n";

} // write_cells_connectivity - 3d

/**
 * Write VTK unstructured grid cells data - 2D.
 */
template<int N>
void write_cells_data(std::ostream& outFile,
		      DataArray2d::HostMirror Uhost,
		      HydroParams& params,
		      ConfigMap& configMap,
		      const std::map<int, std::string>& variables_names,
		      uint64_t& offsetBytes)
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
	      
	      real_t data = Uhost(gw+i,gw+j, sdm::DofMap<2,N>(idx,idy, 0, iVar)); 
	      
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
  
} // write_cells_data - 2D


/**
 * Write VTK unstructured grid cells data - 3D.
 */
template<int N>
void write_cells_data(std::ostream& outFile,
		      DataArray3d::HostMirror Uhost,
		      HydroParams& params,
		      ConfigMap& configMap,
		      const std::map<int, std::string>& variables_names,
		      uint64_t& offsetBytes)
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

		  real_t data = Uhost(gw+i,gw+j, gw+k, sdm::DofMap<3,N>(idx,idy,idz, iVar));
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
  
} // write_cells_data - 3d


// ================================================================
// ================================================================
/**
 * Write VTK unstructured grid binary appended data - 2D.
 */
template<int N>
void write_appended_binary_data(std::ostream& outFile,
				DataArray2d::HostMirror Uhost,
				sdm::SDM_Geometry<2,N> sdm_geom,
				HydroParams& params,
				ConfigMap& configMap,
				const std::map<int, std::string>& variables_names)
{

  UNUSED(configMap);
  
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

  const int nbvar = variables_names.size();

  int nbNodesPerCell = (N+1)*(N+1); // in 2D
  int nbSubCells = N*N;

  outFile << " <AppendedData encoding=\"raw\">" << "\n";

  // leading underscore
  outFile << "_";
  
  /*
   * Write nodes location.
   */
  {
    // this is only necessary for binary output
    std::vector<float> vertices;

    for (int j=0; j<ny; ++j) {
      for (int i=0; i<nx; ++i) {
	
	// cell offset
	real_t xo = xmin + (i+nx*i_mpi)*dx;
	real_t yo = ymin + (j+ny*j_mpi)*dy;
	
	for (int idy=0; idy<N+1; ++idy) {
	  for (int idx=0; idx<N+1; ++idx) {
	    
	    float x, y;
	    
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
	    
	    vertices.push_back(x);
	    vertices.push_back(y);
	    vertices.push_back(0.0);

	  } // for idx
	} // for idy
	
      } // end for i
    } // end for j

    uint64_t size = sizeof(float)*nx*ny*(N+1)*(N+1)*3;
    outFile.write(reinterpret_cast<char *>( &size), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(vertices[0]) ), size);

    vertices.clear();
    
  } // end write nodes location
  
  /*
   * Write connectivity.
   */
  {
    // this is only necessary for binary output
    std::vector<uint64_t> connectivity;

    // cell index
    uint64_t cell_index = 0;
    
    for (int j=0; j<ny; ++j) {
      for (int i=0; i<nx; ++i) {
	
	uint64_t index = i+nx*j;
	
	// offset to the first nodes in this cell
	uint64_t offset = index * nbNodesPerCell;
	
	// loop over sub-cells
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {
	    
	    uint64_t i0,i1,i2,i3;
	    i0 = offset+idx+  (N+1)* idy;
	    i1 = offset+idx+1+(N+1)* idy;
	    i2 = offset+idx+1+(N+1)*(idy+1);
	    i3 = offset+idx  +(N+1)*(idy+1);

	    connectivity.push_back(i0);
	    connectivity.push_back(i1);
	    connectivity.push_back(i2);
	    connectivity.push_back(i3);
	    	    
	  } // for idx
	} // for idy
	
	cell_index += nbSubCells;
	
      } // for i
    } // for j

    uint64_t size = sizeof(uint64_t)*nx*ny*N*N*4;
    outFile.write(reinterpret_cast<char *>( &size), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(connectivity[0]) ), size);
    
    connectivity.clear();
  
  } // end write connectivity
  
  /*
   * Write offsets.
   */
  {
    std::vector<uint64_t> offsets;

    // number of nodes per cell is 4 in 2D
    for (int64_t i=1; i<=nx*ny*N*N; ++i) {
      offsets.push_back(4*i);
    }

    uint64_t size = sizeof(uint64_t)*nx*ny*N*N;
    outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(offsets[0]) ), size);
    offsets.clear();
  
  } // end write offsets
  
  /*
   * Write cell types.
   */
  {
    std::vector<unsigned char> celltypes;

    // 9 means "Quad" - 12 means "Hexahedron"
    for (int64_t i=0; i<nx*ny*N*N; ++i) {
      celltypes.push_back(9);
    }
    
    uint64_t size = sizeof(unsigned char)*nx*ny*N*N;
    outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(celltypes[0]) ), size);
    celltypes.clear();
    
  }

  /*
   * Write cells data.
   */
  {
    const int gw = params.ghostWidth;
    
    std::vector<real_t> cells_data;

    // loop over scalar variables
    for ( int iVar=0; iVar<nbvar; iVar++) {

      // no ghost !!
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  // loop over sub-cells
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {
	      
	      real_t data = Uhost(gw+i,gw+j, sdm::DofMap<2,N>(idx,idy, 0, iVar)); 
	      cells_data.push_back(data);
	      
	    } // for idx
	  } // for idy
	  
	} // for i
      } // for j


      uint64_t size = sizeof(real_t)*nx*ny*N*N;
      outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
      outFile.write(reinterpret_cast<char *>( &(cells_data[0]) ), size);
      cells_data.clear();

    } // end for iVar
    
  } // end write cells data

  outFile << " </AppendedData>" << "\n";
  
} // write_appended_binary_data - 2D

// ================================================================
// ================================================================
/**
 * Write VTK unstructured grid binary appended data - 3D.
 */
template<int N>
void write_appended_binary_data(std::ostream& outFile,
				DataArray3d::HostMirror Uhost,
				sdm::SDM_Geometry<3,N> sdm_geom,
				HydroParams& params,
				ConfigMap& configMap,
				const std::map<int, std::string>& variables_names)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const real_t xmin = params.xmin;
  const real_t ymin = params.ymin;
  const real_t zmin = params.zmin;

  const real_t dx = params.dx;
  const real_t dy = params.dy;
  const real_t dz = params.dz;
  
#ifdef USE_MPI
  const int i_mpi = params.myMpiPos[IX];
  const int j_mpi = params.myMpiPos[IY];
  const int k_mpi = params.myMpiPos[IZ];
#else
  const int i_mpi = 0;
  const int j_mpi = 0;
  const int k_mpi = 0;
#endif

  const int nbvar = variables_names.size();

  int nbNodesPerCell = (N+1)*(N+1)*(N+1); // in 3D
  int nbSubCells = N*N*N;

  int N1=N+1;
  int N2=(N+1)*(N+1);

  outFile << " <AppendedData encoding=\"raw\">" << "\n";

  // leading underscore
  outFile << "_";
  
  /*
   * Write nodes location.
   */
  {

    // this is only necessary for binary output
    std::vector<float> vertices;

    for (int k=0; k<nz; ++k) {
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  // cell offset
	  real_t xo = xmin + (i+nx*i_mpi)*dx;
	  real_t yo = ymin + (j+ny*j_mpi)*dy;
	  real_t zo = zmin + (k+nz*k_mpi)*dz;
	  
	  for (int idz=0; idz<N+1; ++idz) {
	    for (int idy=0; idy<N+1; ++idy) {
	      for (int idx=0; idx<N+1; ++idx) {
		
		real_t x,y,z;
		
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
		
		if (idz == 0) {
		  z = zo;
		} else if ( idz == N) {
		  z = zo + dz;
		} else { 
		  
		  z = zo + 0.5 * (sdm_geom.solution_pts_1d_host(idz-1) +
				  sdm_geom.solution_pts_1d_host(idz  ) ) * dz;
		}
		
		vertices.push_back(x);
		vertices.push_back(y);
		vertices.push_back(z);
		  
	      } // for idx
	    } // for idy
	  } // for idz
	  
	} // end for i
      } // end for j
    } // end for k

    uint64_t size = sizeof(float)*nx*ny*nz*(N+1)*(N+1)*(N+1)*3;
    outFile.write(reinterpret_cast<char *>( &size), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(vertices[0])), size);

    vertices.clear();
  }

  /*
   * Write connectivity.
   */
  {
    // this is only necessary for binary output
    std::vector<uint64_t> connectivity;

    // cell index
    uint64_t cell_index = 0;

    for (int k=0; k<nz; ++k) {
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  uint64_t index = i+nx*j+nx*ny*k;
	  
	  // offset to the first nodes in this cell
	  uint64_t offset = index * nbNodesPerCell;
	  
	  // loop over sub-cells
	  for (int idz=0; idz<N; ++idz) {
	    for (int idy=0; idy<N; ++idy) {
	      for (int idx=0; idx<N; ++idx) {
		
		connectivity.push_back(offset+idx  +N1* idy   + N2* idz   );
		connectivity.push_back(offset+idx+1+N1* idy   + N2* idz   );
		connectivity.push_back(offset+idx+1+N1*(idy+1)+ N2* idz   );
		connectivity.push_back(offset+idx  +N1*(idy+1)+ N2* idz   );
		connectivity.push_back(offset+idx  +N1* idy   + N2*(idz+1));
		connectivity.push_back(offset+idx+1+N1* idy   + N2*(idz+1));
		connectivity.push_back(offset+idx+1+N1*(idy+1)+ N2*(idz+1));
		connectivity.push_back(offset+idx  +N1*(idy+1)+ N2*(idz+1));
		
	      } // for idx
	    } // for idy
	  } // for idz
	  
	  cell_index += nbSubCells;
	  
	} // for i
      } // for j
    } // for k

    uint64_t size = sizeof(uint64_t)*nx*ny*nz*N*N*N*8;
    outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(connectivity[0]) ), size);
    
    connectivity.clear();
    
  } // end write connectivity
  
  /*
   * Write offsets.
   */
  {
    std::vector<uint64_t> offsets;

    // number of nodes per cell is 8 in 3D
    for (int64_t i=1; i<=nx*ny*nz*N*N*N; ++i) {
      offsets.push_back(8*i);
    }

    uint64_t size = sizeof(uint64_t)*nx*ny*nz*N*N*N;
    outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(offsets[0]) ), size);
    offsets.clear();

  }

  /*
   * Write cell types.
   */
  {
    std::vector<unsigned char> celltypes;
    
    // 9 means "Quad" - 12 means "Hexahedron"
    for (int64_t i=0; i<nx*ny*nz*N*N*N; ++i) {
      celltypes.push_back(12);
    }
    
    uint64_t size = sizeof(unsigned char)*nx*ny*nz*N*N*N;
    outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
    outFile.write(reinterpret_cast<char *>( &(celltypes[0]) ), size);
    celltypes.clear();
    
  }

  /*
   * Write cells data.
   */
  {
    const int gw = params.ghostWidth;
    
    std::vector<real_t> cells_data;

    // loop over scalar variables
    for ( int iVar=0; iVar<nbvar; iVar++) {

      // no ghost !!
      for (int k=0; k<nz; ++k) {
	for (int j=0; j<ny; ++j) {
	  for (int i=0; i<nx; ++i) {
	    
	    // loop over sub-cells
	    for (int idz=0; idz<N; ++idz) {
	      for (int idy=0; idy<N; ++idy) {
		for (int idx=0; idx<N; ++idx) {
		  
		  real_t data = Uhost(gw+i,gw+j, gw+k, sdm::DofMap<3,N>(idx,idy,idz, iVar));

		  cells_data.push_back(data);
		  
		} // for idx
	      } // for idy
	    } // for idz
	  
	  } // for i
	} // for j
      } // for k

      uint64_t size = sizeof(real_t)*nx*ny*nz*N*N*N;
      outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
      outFile.write(reinterpret_cast<char *>( &(cells_data[0]) ), size);
      cells_data.clear();
      
    } // end for iVar
    
  } // end write cells data

  outFile << " </AppendedData>" << "\n";

} // write_appended_binary_data - 3D

#ifdef USE_MPI
// ================================================================
// ================================================================
/**
 * Write Parallel VTU header. 
 * Must be done by a single MPI process.
 *
 * \note optionnal parameter is_flux_data_array when true is used to 
 * trigger saving a flux data array, for which dof are attached to nodes
 * rather than cells.
 *
 */
// void write_pvtu_header(std::string headerFilename,
// 		       std::string outputPrefix,
// 		       HydroParams& params,
// 		       ConfigMap& configMap,
// 		       int nbvar,
// 		       const std::map<int, std::string>& varNames,
// 		       int iStep,
// 		       bool is_flux_data_array = false);
#endif // USE_MPI

// ================================================================
// ================================================================
/**
 * 2D Output routine (VTK file format, ASCII, VtkUnstructuredGrid)
 * for High-Order Spectral Difference method schemes.

 * We use UnstructuredGrid here because, the mesh cells are unevenly
 * split inti subcells, one per Dof of the SDM scheme.
 *
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
template<int N>
void save_VTK_SDM(DataArray2d             Udata,
		  DataArray2d::HostMirror Uhost,
		  HydroParams& params,
		  ConfigMap& configMap,
		  sdm::SDM_Geometry<2,N> sdm_geom,
		  int nbvar,
		  const std::map<int, std::string>& variables_names,
		  int iStep,
		  real_t time,
		  std::string debug_name = "")
{
  UNUSED(nbvar);

  const int nx = params.nx;
  const int ny = params.ny;

  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  
  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;

#ifdef USE_MPI
  // write pvtu wrapper file
  if (params.myRank == 0) {
    
    // header file : parallel vtu format
    std::string headerFilename = outputDir+"/"+outputPrefix+"_time"+stepNum.str()+".pvtu";

    if ( !debug_name.empty() )
      headerFilename = outputDir+"/"+outputPrefix+"_"+debug_name+"_time"+stepNum.str()+".pvtu";
    
    write_pvtu_header(headerFilename,
		      outputPrefix,
		      params,
		      configMap,
		      nbvar,
		      variables_names,
		      iStep);
  }
#endif // USE_MPI

  // concatenate file prefix + file number + suffix
  std::string filename;
  filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vtu";

  if ( !debug_name.empty() )
    filename = outputDir + "/" + outputPrefix + "_" + debug_name + "_" + stepNum.str() + ".vtu";

#ifdef USE_MPI
  {
    // write MPI rank in string rankFormat
    std::ostringstream rankFormat;
    rankFormat.width(5);
    rankFormat.fill('0');
    rankFormat << params.myRank;
    
    // modify filename for mpi
    filename = outputDir + "/" + outputPrefix + "_time" + stepNum.str()+"_mpi"+rankFormat.str()+".vtu";
  }
#endif // USE_MPI  
  
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

  /*
   * write nodes location + data.
   */
  uint64_t offsetBytes = 0;

  write_nodes_location<N>(outFile,sdm_geom,params,configMap,offsetBytes);

  write_cells_connectivity<N>(outFile,sdm_geom,params,configMap,offsetBytes);

  write_cells_data<N>(outFile,Uhost,params,configMap,variables_names,offsetBytes);
  
  outFile << " </Piece>\n";
  
  outFile << " </UnstructuredGrid>\n";

  // write appended binary data (no compression, just raw binary)
  if (!outputVtkAscii)
    write_appended_binary_data(outFile, Uhost, sdm_geom, params, configMap, variables_names);
  
  outFile << "</VTKFile>\n";
  
  outFile.close();
  
} // end save_VTK_SDM<N> - 2D

// ================================================================
// ================================================================
/**
 * 3D Output routine (VTK file format, ASCII, VtkUnstructuredGrid)
 * for High-Order Spectral Difference method schemes.

 * We use UnstructuredGrid here because, the mesh cells are unevenly
 * split inti subcells, one per Dof of the SDM scheme.
 *
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
template<int N>
void save_VTK_SDM(DataArray3d             Udata,
		  DataArray3d::HostMirror Uhost,
		  HydroParams& params,
		  ConfigMap& configMap,
		  sdm::SDM_Geometry<3,N> sdm_geom,
		  int nbvar,
		  const std::map<int, std::string>& variables_names,
		  int iStep,
		  real_t time,
		  std::string debug_name = "")
{
  UNUSED(nbvar);

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  
  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;
  
#ifdef USE_MPI
  // write pvtu wrapper file
  if (params.myRank == 0) {
    
    // header file : parallel vtu format
    std::string headerFilename = outputDir+"/"+outputPrefix+"_time"+stepNum.str()+".pvtu";
    
    if ( !debug_name.empty() )
      headerFilename = outputDir+"/"+outputPrefix+"_"+debug_name+"_time"+stepNum.str()+".pvtu";

    write_pvtu_header(headerFilename,
		      outputPrefix,
		      params,
		      configMap,
		      nbvar,
		      variables_names,
		      iStep);
  }
#endif // USE_MPI

  // concatenate file prefix + file number + suffix
  std::string filename;
  filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vtu";

  if ( !debug_name.empty() )
    filename = outputDir + "/" + outputPrefix + "_" + debug_name + "_" + stepNum.str() + ".vtu";

#ifdef USE_MPI
  {
    // write MPI rank in string rankFormat
    std::ostringstream rankFormat;
    rankFormat.width(5);
    rankFormat.fill('0');
    rankFormat << params.myRank;

    // modify filename for mpi
    filename = outputDir + "/" + outputPrefix + "_time" + stepNum.str()+"_mpi"+rankFormat.str()+".vtu";
  }
#endif // USE_MPI  

  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header
  write_vtu_header(outFile, configMap);

  // write vtk metadata (time and iStep)
  write_vtk_metadata(outFile, iStep, time);
  
  // write mesh information
  // each "cell" actually has a N^2 mini-mesh
  int nbOfCells = nx*ny*nz *  N   * N   * N;
  int nbOfNodes = nx*ny*nz * (N+1)*(N+1)*(N+1);
  
  outFile << "<Piece NumberOfPoints=\"" << nbOfNodes
	  <<"\" NumberOfCells=\"" << nbOfCells << "\" >\n";
  
  /*
   * write nodes location + data.
   */
  uint64_t offsetBytes = 0;
  
  write_nodes_location<N>(outFile,sdm_geom,params,configMap,offsetBytes);
  
  write_cells_connectivity<N>(outFile,sdm_geom,params,configMap,offsetBytes);
  
  write_cells_data<N>(outFile,Uhost,params,configMap,variables_names,offsetBytes);
  
  outFile << " </Piece>\n";
  
  outFile << " </UnstructuredGrid>\n";
  
  // write appended binary data (no compression, just raw binary)
  if (!outputVtkAscii)
    write_appended_binary_data(outFile, Uhost, sdm_geom, params, configMap, variables_names);
  
  outFile << "</VTKFile>\n";

  outFile.close();
  
} // end save_VTK_SDM<N> - 3D

} // namespace io

} // namespace ppkMHD

#endif // IO_VTK_SDM_H_
