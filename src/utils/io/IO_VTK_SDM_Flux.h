/**
 * Some specialized output routines to dump data in VTK unstructured grid
 * format for the High-order Spectral Difference Method schemes.
 * 
 * In this file we provide a slightly different version for flux data arrays.
 */

#ifndef IO_VTK_SDM_FLUX_H_
#define IO_VTK_SDM_FLUX_H_

#include <map>
#include <string>

#include <cstdint>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "utils/io/IO_VTK_SDM_shared.h"

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid nodes locations (x,y,z).
 *
 * \param[in,out] offsetBytes is incremented by the size of data written (only
 *                useful for appended binary data).
 *
 * This routine is a specialized version of write_nodes_location for the
 * case we want to dump data at flux points.
 *
 * \note we enlarge the flux points set in the transverse direction, so that
 * the sub cell borders match the cell border.
 *
 * \tparam dir specifies the flux direction
 */
template<int N, int dir>
void write_nodes_location_flux(std::ostream& outFile,
			       DataArray2d::HostMirror Uhost,
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
  
  const int ghostWidth = params.ghostWidth;

  bool useDouble = sizeof(real_t) == sizeof(double) ? true : false;
  const char* dataType = useDouble ? "Float64" : "Float32";

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  int nbNodesPerCell = (N+1)*(N+2); // in 3D
  
  // bounds used to loop over sub-nodes
  // idx_end x idy_end is equal to nbNodesPerCells
  int idx_end;
  int idy_end;

  if (dir == IX) {
    // nodes     layout N+1 x N+2
    // sub-cells layout N   x N+1
    idx_end = N+1;
    idy_end = N+2;
  } else if (dir == IY) {
    // nodes     layout N+2 x N+1
    // sub-cells layout N+1 x N  
    idx_end = N+2;
    idy_end = N+1;
  }

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
	
	for (int idy=0; idy<idy_end; ++idy) {
	  
	  float y;
	  if (dir == IY) {
	    y = yo + sdm_geom.flux_pts_1d_host(idy) * dy;
	  } else {
	    if (idy == 0) {
	      y = yo;
	    } else if ( idy == N+1) {
	      y = yo + dy;
	    } else { 
	      y = yo + sdm_geom.solution_pts_1d_host(idy-1) * dy;
	    }
	  }
	  	  
	  for (int idx=0; idx<idx_end; ++idx) {
	    
	    float x;
	    if (dir == IX) {
	      x = xo + sdm_geom.flux_pts_1d_host(idx) * dx;
	    } else {
	      if (idx == 0) {
		x = xo;
	      } else if ( idx == N+1) {
		x = xo + dx;
	      } else { 
		x = xo + sdm_geom.solution_pts_1d_host(idx-1) * dx;
	      }
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

  offsetBytes += sizeof(uint64_t) + sizeof(float)*nx*ny*(N+1)*(N+2)*3;
  
} // write_nodes_location_flux - 2d

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid nodes locations (x,y,z) - 3D.
 *
 * \param[in,out] offsetBytes is incremented by the size of data written (only
 *                useful for appended binary data).
 *
 * This routine is a specialized version of write_nodes_location for the
 * case we want to dump data at flux points.
 *
 * \note we enlarge the flux points set in the transverse direction, so that
 * the sub cell borders match the cell border.
 *
 * \tparam dir specifies the flux direction
 */
template<int N,int dir>
void write_nodes_location_flux(std::ostream& outFile,
			       DataArray3d::HostMirror Uhost,
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
  
  const int ghostWidth = params.ghostWidth;

  bool useDouble = sizeof(real_t) == sizeof(double) ? true : false;
  const char* dataType = useDouble ? "Float64" : "Float32";

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  int nbNodesPerCell = (N+1)*(N+2)*(N+2); // in 3D
  
  // bounds used to loop over sub-nodes
  // idx_end x idy_end x idz_end is equal to nbNodesPerCells
  int idx_end;
  int idy_end;
  int idz_end;

  if (dir == IX) {
    // nodes     layout N+1 x N+2 x N+2
    // sub-cells layout N   x N+1 x N+1
    idx_end = N+1;
    idy_end = N+2;
    idz_end = N+2;
  } else if (dir == IY) {
    // nodes     layout N+2 x N+1 x N+2
    // sub-cells layout N+1 x N   x N+1
    idx_end = N+2;
    idy_end = N+1;
    idz_end = N+2;
  } else { // dir == IZ
    // nodes     layout N+2 x N+2 x N+1
    // sub-cells layout N+1 x N+1 x N
    idx_end = N+2;
    idy_end = N+2;
    idz_end = N+1;
  }

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
	  
	  for (int idz=0; idz<idz_end; ++idz) {
	    
	    float z;
	    if (dir == IZ) {
	      z = zo + sdm_geom.flux_pts_1d_host(idz) * dz;
	    } else {
	      if (idz == 0) {
		z = zo;
	      } else if ( idz == N+1) {
		z = zo + dz;
	      } else { 
		z = zo + sdm_geom.solution_pts_1d_host(idz-1) * dz;
	      }
	    }
	    
	    for (int idy=0; idy<idy_end; ++idy) {
	      
	      float y;
	      if (dir == IY) {
		y = yo + sdm_geom.flux_pts_1d_host(idy) * dy;
	      } else {
		if (idy == 0) {
		  y = yo;
		} else if ( idy == N+1) {
		  y = yo + dy;
		} else { 
		  y = yo + sdm_geom.solution_pts_1d_host(idy-1) * dy;
		}
	      }
	      
	      for (int idx=0; idx<idx_end; ++idx) {
		
		float x;

		if (dir == IX) {
		  x = xo + sdm_geom.flux_pts_1d_host(idx) * dx;
		} else {
		  if (idx == 0) {
		    x = xo;
		  } else if ( idx == N+1) {
		    x = xo + dx;
		  } else { 
		    x = xo + sdm_geom.solution_pts_1d_host(idx-1) * dx;
		  }
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
  
  offsetBytes += sizeof(uint64_t) + sizeof(float)*nx*ny*nz*nbNodesPerCell*3;

} // write_nodes_location - 3d

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid nodes connectivity + offsets + cell type
 * (quad in 2D).
 *
 * returned value is only useful when using appended binary data, the returned
 * is actually the currently value of "offset".
 *
 * This routine is a specialized version of write_cells_connectivity for the
 * case we want to dump data at flux points.
 *
 * \tparam dir specifies the flux direction
 */
template<int N, int dir>
void write_cells_connectivity_flux(std::ostream& outFile,
				   DataArray2d::HostMirror Uhost,
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
  
  const int ghostWidth = params.ghostWidth;

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  int nbNodesPerCell = (N+1)*(N+2); // in 2D
  int nbSubCells = N*(N+1);
  
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

    if (dir == IX) {
      
      // cell index
      int cell_index = 0;
      
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  uint64_t index = i+nx*j;
	  
	  // offset to the first nodes in this cell
	  uint64_t offset = index * nbNodesPerCell;
	  
	  // sub-cells grid  :  N   x N+1
	  // sub-cells nodes :  N+1 x N+2
	  for (int idy=0; idy<N+1; ++idy) {
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

    } // end dir IX
    
    if (dir == IY) {
      
      // cell index
      int cell_index = 0;
      
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  uint64_t index = i+nx*j;
	  
	  // offset to the first nodes in this cell
	  uint64_t offset = index * nbNodesPerCell;
	  
	  // sub-cells grid  :  N+1 x N
	  // sub-cells nodes :  N+2 x N+1
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N+1; ++idx) {
	      
	      uint64_t i0,i1,i2,i3;
	      i0 = offset+idx+  (N+2)* idy;
	      i1 = offset+idx+1+(N+2)* idy;
	      i2 = offset+idx+1+(N+2)*(idy+1);
	      i3 = offset+idx  +(N+2)*(idy+1);
	      
	      outFile << i0 << " " << i1 << " " << i2 << " " << i3 << "\n";
	      
	    } // for idx
	  } // for idy
	  
	  cell_index += nbSubCells;
	  
	} // for i
      } // for j

    } // end dir IY
    
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

  offsetBytes += sizeof(uint64_t) + sizeof(uint64_t)*nx*ny*nbSubCells;

  if (outputVtkAscii) {
    // number of nodes per cell is 4 in 2D
    for (int i=1; i<=nx*ny*nbSubCells; ++i) {
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

  offsetBytes += sizeof(uint64_t) + sizeof(unsigned char)*nx*ny*nbSubCells;
  
  if (outputVtkAscii) {
    // 9 means "Quad" - 12 means "Hexahedron"
    for (int i=0; i<nx*ny*nbSubCells; ++i) {
      outFile << 9 << " ";
    }

    outFile << "\n";
  }
  
  outFile << "    </DataArray>\n";

  /*
   * Close Cells section.
   */
  outFile << "  </Cells>\n";

} // write_cells_connectivity_flux - 2d

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid nodes connectivity + offsets + cell type
 * (hexahedron in 3D).
 *
 * returned value is only useful when using appended binary data, the returned
 * is actually the currently value of "offset".
 *
 * This routine is a specialized version of write_cells_connectivity for the
 * case we want to dump data at flux points.
 *
 * \tparam dir specifies the flux direction
 */
template<int N, int dir>
void write_cells_connectivity_flux(std::ostream& outFile,
				   DataArray3d::HostMirror Uhost,
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
  
  const int ghostWidth = params.ghostWidth;

  bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

  int nbNodesPerCell = (N+1)*(N+2)*(N+2); // in 3D
  int nbSubCells = N*(N+1)*(N+1);
    
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

  // bounds used to loop over sub-cells
  // idx_end x idy_end x idz_end is equal to nbSubCells
  int idx_end;
  int idy_end;
  int idz_end;

  if (dir == IX) {
    // nodes     layout N+1 x N+2 x N+2
    // sub-cells layout N   x N+1 x N+1
    idx_end = N;
    idy_end = N+1;
    idz_end = N+1;
  } else if (dir == IY) {
    // nodes     layout N+2 x N+1 x N+2
    // sub-cells layout N+1 x N   x N+1
    idx_end = N+1;
    idy_end = N;
    idz_end = N+1;
  } else { // dir == IZ
    // nodes     layout N+2 x N+2 x N+1
    // sub-cells layout N+1 x N+1 x N
    idx_end = N+1;
    idy_end = N+1;
    idz_end = N;
  }

  // constants used to linearize index to sub-cells nodes
  int N1 =  idx_end+1;
  int N2 = (idx_end+1)*(idy_end+1);

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
	  for (int idz=0; idz<idz_end; ++idz) {
	    for (int idy=0; idy<idy_end; ++idy) {
	      for (int idx=0; idx<idx_end; ++idx) {
		
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

} // write_cells_connectivity_flux - 3d

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid - flux point data - 2D.
 * 
 * Remember we write point data (not cell data). 
 *
 * \tparam dir specifies the flux direction
 */
template<int N,int dir>
void write_flux_points_data(std::ostream& outFile,
			    DataArray2d::HostMirror Uhost,
			    sdm::SDM_Geometry<2,N> sdm_geom,
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

  int nbNodesPerCell = (N+1)*(N+2); // in 2D
  
  // bounds used to loop over sub-nodes
  // idx_end x idy_end is equal to nbNodesPerCells
  int idx_end;
  int idy_end;

  if (dir == IX) {
    // nodes     layout N+1 x N+2
    // sub-cells layout N   x N+1
    idx_end = N+1;
    idy_end = N+2;
  } else if (dir == IY) {
    // nodes     layout N+2 x N+1
    // sub-cells layout N+1 x N  
    idx_end = N+2;
    idy_end = N+1;
  }

  /*
   * write point data with flux at flux point.
   */
  outFile << "  <PointData>\n";

  // loop over scalar variables
  for ( int iVar=0; iVar<params.nbvar; iVar++ ) {
    
    outFile << "    <DataArray type=\"" << dataType
	    << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"" << ascii_or_binary << "\"";
    
    if (!outputVtkAscii) {
      outFile << " offset=\"" << offsetBytes << "\"";
    }

    outFile << " >\n";
    
    offsetBytes += sizeof(uint64_t) + sizeof(real_t)*nx*ny*nbNodesPerCell;
    
    if (outputVtkAscii) {

      // no ghost !!
      for (int j=0; j<ny; ++j) {
	for (int i=0; i<nx; ++i) {
	  
	  // loop over sub-nodes
	  for (int idy=0; idy<idy_end; ++idy) {
	    for (int idx=0; idx<idx_end; ++idx) {
	      
	      int iidx = idx;
	      int iidy = idy;
	      
	      if (dir == IX) {

		if (idy == 0)
		  iidy = 1;
		else if (idy == N+1)
		  iidy = N;
		iidy -= 1;
		  		
	      } else { // dir == IY

		if (idx == 0)
		  iidx = 1;
		else if (idx == N+1)
		  iidx = N;
		iidx -= 1;

	      }
	      
	      real_t data = Uhost(gw+i,gw+j, sdm::DofMapFlux<2,N,dir>(iidx,iidy, 0, iVar));
	      
	      outFile << data << " ";
	      
	    } // for idx
	  } // for idy
	  
	} // for i
      } // for j
      
      outFile << "\n";

    } // end outputVtkAscii
    
    outFile << "    </DataArray>\n";
    
  } // end for variables
  
  outFile << "  </PointData>\n";
  
} // write_flux_points_data - 2D

// =======================================================
// =======================================================
/**
 * Write VTK unstructured grid cells data - 3D.
 */
template<int N, int dir>
void write_flux_points_data(std::ostream& outFile,
			    DataArray3d::HostMirror Uhost,
			    sdm::SDM_Geometry<3,N> sdm_geom,
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

  int nbNodesPerCell = (N+1)*(N+2)*(N+2); // in 3D
  
  // bounds used to loop over sub-nodes
  // idx_end x idy_end x idz_end is equal to nbNodesPerCells
  int idx_end;
  int idy_end;
  int idz_end;

  if (dir == IX) {
    // nodes     layout N+1 x N+2 x N+2
    // sub-cells layout N   x N+1 x N+1
    idx_end = N+1;
    idy_end = N+2;
    idz_end = N+2;
  } else if (dir == IY) {
    // nodes     layout N+2 x N+1 x N+2
    // sub-cells layout N+1 x N   x N+1
    idx_end = N+2;
    idy_end = N+1;
    idz_end = N+2;
  } else { // dir == IZ
    // nodes     layout N+2 x N+2 x N+1
    // sub-cells layout N+1 x N+1 x N
    idx_end = N+2;
    idy_end = N+2;
    idz_end = N+1;
  }

  /*
   * write cell data.
   */
  outFile << "  <PointData>\n";

  // loop over scalar variables
  for ( int iVar=0; iVar<params.nbvar; iVar++ ) {
    
    outFile << "    <DataArray type=\"" << dataType
	    << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"" << ascii_or_binary << "\"";

    if (!outputVtkAscii) {
      outFile << " offset=\"" << offsetBytes << "\"";
    }

    outFile<< " >\n";

    offsetBytes += sizeof(uint64_t) + sizeof(real_t)*nx*ny*nz*nbNodesPerCell;

    if (outputVtkAscii) {
      // no ghost !!
      for (int k=0; k<nz; ++k) {
	for (int j=0; j<ny; ++j) {
	  for (int i=0; i<nx; ++i) {
	    
	    // loop over sub-cells
	    for (int idz=0; idz<idz_end; ++idz) {
	      for (int idy=0; idy<idy_end; ++idy) {
		for (int idx=0; idx<idx_end; ++idx) {

		  int iidx = idx;
		  int iidy = idy;
		  int iidz = idz;
		  
		  if (dir == IX) {
		    
		    if (idy == 0)
		      iidy = 1;
		    else if (idy == N+1)
		      iidy = N;
		    iidy -= 1;

		    if (idz == 0)
		      iidz = 1;
		    else if (idz == N+1)
		      iidz = N;
		    iidz -= 1;

		  } else if (dir == IY) {
		    
		    if (idx == 0)
		      iidx = 1;
		    else if (idx == N+1)
		      iidx = N;
		    iidx -= 1;

		    if (idz == 0)
		      iidz = 1;
		    else if (idz == N+1)
		      iidz = N;
		    iidz -= 1;

		  } else {  // dir == IZ

		    if (idx == 0)
		      iidx = 1;
		    else if (idx == N+1)
		      iidx = N;
		    iidx -= 1;
		    
		    if (idy == 0)
		      iidy = 1;
		    else if (idy == N+1)
		      iidy = N;
		    iidy -= 1;

		  }

		  real_t data = Uhost(gw+i,gw+j,gw+k,
				      sdm::DofMapFlux<3,N,dir>(iidx,iidy,iidz,
							       iVar));
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
  
  outFile << "  </PointData>\n";
  
} // write_flux_points_data - 3d


// // // ================================================================
// // // ================================================================
// // /**
// //  * Write VTK unstructured grid binary appended data - 2D.
// //  */
// // template<int N>
// // void write_appended_binary_data(std::ostream& outFile,
// // 				DataArray2d::HostMirror Uhost,
// // 				sdm::SDM_Geometry<2,N> sdm_geom,
// // 				HydroParams& params,
// // 				ConfigMap& configMap,
// // 				const std::map<int, std::string>& variables_names)
// // {

// //   const int nx = params.nx;
// //   const int ny = params.ny;

// //   const real_t xmin = params.xmin;
// //   const real_t ymin = params.ymin;

// //   const real_t dx = params.dx;
// //   const real_t dy = params.dy;
  
// // #ifdef USE_MPI
// //   const int i_mpi = params.myMpiPos[IX];
// //   const int j_mpi = params.myMpiPos[IY];
// // #else
// //   const int i_mpi = 0;
// //   const int j_mpi = 0;
// // #endif

// //   int nbNodesPerCell = (N+1)*(N+1); // in 2D
// //   int nbSubCells = N*N;

// //   outFile << " <AppendedData encoding=\"raw\">" << "\n";

// //   // leading underscore
// //   outFile << "_";
  
// //   /*
// //    * Write nodes location.
// //    */
// //   {
// //     // this is only necessary for binary output
// //     std::vector<float> vertices;

// //     for (int j=0; j<ny; ++j) {
// //       for (int i=0; i<nx; ++i) {
	
// // 	// cell offset
// // 	real_t xo = xmin + (i+nx*i_mpi)*dx;
// // 	real_t yo = ymin + (j+ny*j_mpi)*dy;
	
// // 	for (int idy=0; idy<N+1; ++idy) {
// // 	  for (int idx=0; idx<N+1; ++idx) {
	    
// // 	    float x, y;
	    
// // 	    if (idx == 0) {
// // 	      x = xo;
// // 	    } else if ( idx == N) {
// // 	      x = xo + dx;
// // 	    } else { 
	      
// // 	      x = xo + 0.5 * (sdm_geom.solution_pts_1d_host(idx-1) +
// // 			      sdm_geom.solution_pts_1d_host(idx)   ) * dx;
	      
// // 	    }
	    
// // 	    if (idy == 0) {
// // 	      y = yo;
// // 	    } else if ( idy == N) {
// // 	      y = yo + dy;
// // 	    } else { 
	      
// // 	      y = yo + 0.5 * (sdm_geom.solution_pts_1d_host(idy-1) +
// // 			      sdm_geom.solution_pts_1d_host(idy  ) ) * dy;
// // 	    }
	    
// // 	    vertices.push_back(x);
// // 	    vertices.push_back(y);
// // 	    vertices.push_back(0.0);

// // 	  } // for idx
// // 	} // for idy
	
// //       } // end for i
// //     } // end for j

// //     uint64_t size = sizeof(float)*nx*ny*(N+1)*(N+1)*3;
// //     outFile.write(reinterpret_cast<char *>( &size), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(vertices[0]) ), size);

// //     vertices.clear();
    
// //   } // end write nodes location
  
// //   /*
// //    * Write connectivity.
// //    */
// //   {
// //     // this is only necessary for binary output
// //     std::vector<uint64_t> connectivity;

// //     // cell index
// //     uint64_t cell_index = 0;
    
// //     for (int j=0; j<ny; ++j) {
// //       for (int i=0; i<nx; ++i) {
	
// // 	uint64_t index = i+nx*j;
	
// // 	// offset to the first nodes in this cell
// // 	uint64_t offset = index * nbNodesPerCell;
	
// // 	// loop over sub-cells
// // 	for (int idy=0; idy<N; ++idy) {
// // 	  for (int idx=0; idx<N; ++idx) {
	    
// // 	    uint64_t i0,i1,i2,i3;
// // 	    i0 = offset+idx+  (N+1)* idy;
// // 	    i1 = offset+idx+1+(N+1)* idy;
// // 	    i2 = offset+idx+1+(N+1)*(idy+1);
// // 	    i3 = offset+idx  +(N+1)*(idy+1);

// // 	    connectivity.push_back(i0);
// // 	    connectivity.push_back(i1);
// // 	    connectivity.push_back(i2);
// // 	    connectivity.push_back(i3);
	    	    
// // 	  } // for idx
// // 	} // for idy
	
// // 	cell_index += nbSubCells;
	
// //       } // for i
// //     } // for j

// //     uint64_t size = sizeof(uint64_t)*nx*ny*N*N*4;
// //     outFile.write(reinterpret_cast<char *>( &size), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(connectivity[0]) ), size);
    
// //     connectivity.clear();
  
// //   } // end write connectivity
  
// //   /*
// //    * Write offsets.
// //    */
// //   {
// //     std::vector<uint64_t> offsets;

// //     // number of nodes per cell is 4 in 2D
// //     for (uint64_t i=1; i<=nx*ny*N*N; ++i) {
// //       offsets.push_back(4*i);
// //     }

// //     uint64_t size = sizeof(uint64_t)*nx*ny*N*N;
// //     outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(offsets[0]) ), size);
// //     offsets.clear();
  
// //   } // end write offsets
  
// //   /*
// //    * Write cell types.
// //    */
// //   {
// //     std::vector<unsigned char> celltypes;

// //     // 9 means "Quad" - 12 means "Hexahedron"
// //     for (uint64_t i=0; i<nx*ny*N*N; ++i) {
// //       celltypes.push_back(9);
// //     }
    
// //     uint64_t size = sizeof(unsigned char)*nx*ny*N*N;
// //     outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(celltypes[0]) ), size);
// //     celltypes.clear();
    
// //   }

// //   /*
// //    * Write cells data.
// //    */
// //   {
// //     const int gw = params.ghostWidth;
    
// //     std::vector<real_t> cells_data;

// //     // loop over scalar variables
// //     for ( int iVar=0; iVar<params.nbvar; iVar++) {

// //       // no ghost !!
// //       for (int j=0; j<ny; ++j) {
// // 	for (int i=0; i<nx; ++i) {
	  
// // 	  // loop over sub-cells
// // 	  for (int idy=0; idy<N; ++idy) {
// // 	    for (int idx=0; idx<N; ++idx) {
	      
// // 	      real_t data = Uhost(gw+i,gw+j, sdm::DofMap<2,N>(idx,idy, 0, iVar)); 
// // 	      cells_data.push_back(data);
	      
// // 	    } // for idx
// // 	  } // for idy
	  
// // 	} // for i
// //       } // for j


// //       uint64_t size = sizeof(real_t)*nx*ny*N*N;
// //       outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
// //       outFile.write(reinterpret_cast<char *>( &(cells_data[0]) ), size);
// //       cells_data.clear();

// //     } // end for iVar
    
// //   } // end write cells data

// //   outFile << " </AppendedData>" << "\n";
  
// // } // write_appended_binary_data - 2D

// // // ================================================================
// // // ================================================================
// // /**
// //  * Write VTK unstructured grid binary appended data - 3D.
// //  */
// // template<int N>
// // void write_appended_binary_data(std::ostream& outFile,
// // 				DataArray3d::HostMirror Uhost,
// // 				sdm::SDM_Geometry<3,N> sdm_geom,
// // 				HydroParams& params,
// // 				ConfigMap& configMap,
// // 				const std::map<int, std::string>& variables_names)
// // {

// //   const int nx = params.nx;
// //   const int ny = params.ny;
// //   const int nz = params.nz;

// //   const real_t xmin = params.xmin;
// //   const real_t ymin = params.ymin;
// //   const real_t zmin = params.zmin;

// //   const real_t dx = params.dx;
// //   const real_t dy = params.dy;
// //   const real_t dz = params.dz;
  
// // #ifdef USE_MPI
// //   const int i_mpi = params.myMpiPos[IX];
// //   const int j_mpi = params.myMpiPos[IY];
// //   const int k_mpi = params.myMpiPos[IZ];
// // #else
// //   const int i_mpi = 0;
// //   const int j_mpi = 0;
// //   const int k_mpi = 0;
// // #endif

// //   int nbNodesPerCell = (N+1)*(N+1)*(N+1); // in 3D
// //   int nbSubCells = N*N*N;

// //   int N1=N+1;
// //   int N2=(N+1)*(N+1);

// //   outFile << " <AppendedData encoding=\"raw\">" << "\n";

// //   // leading underscore
// //   outFile << "_";
  
// //   /*
// //    * Write nodes location.
// //    */
// //   {

// //     // this is only necessary for binary output
// //     std::vector<float> vertices;

// //     for (int k=0; k<nz; ++k) {
// //       for (int j=0; j<ny; ++j) {
// // 	for (int i=0; i<nx; ++i) {
	  
// // 	  // cell offset
// // 	  real_t xo = xmin + (i+nx*i_mpi)*dx;
// // 	  real_t yo = ymin + (j+ny*j_mpi)*dy;
// // 	  real_t zo = zmin + (k+nz*k_mpi)*dz;
	  
// // 	  for (int idz=0; idz<N+1; ++idz) {
// // 	    for (int idy=0; idy<N+1; ++idy) {
// // 	      for (int idx=0; idx<N+1; ++idx) {
		
// // 		real_t x,y,z;
		
// // 		if (idx == 0) {
// // 		  x = xo;
// // 		} else if ( idx == N) {
// // 		  x = xo + dx;
// // 		} else { 
		  
// // 		  x = xo + 0.5 * (sdm_geom.solution_pts_1d_host(idx-1) +
// // 				  sdm_geom.solution_pts_1d_host(idx)   ) * dx;
		  
// // 		}
		
// // 		if (idy == 0) {
// // 		  y = yo;
// // 		} else if ( idy == N) {
// // 		  y = yo + dy;
// // 		} else { 
		  
// // 		  y = yo + 0.5 * (sdm_geom.solution_pts_1d_host(idy-1) +
// // 				  sdm_geom.solution_pts_1d_host(idy  ) ) * dy;
// // 		}
		
// // 		if (idz == 0) {
// // 		  z = zo;
// // 		} else if ( idz == N) {
// // 		  z = zo + dz;
// // 		} else { 
		  
// // 		  z = zo + 0.5 * (sdm_geom.solution_pts_1d_host(idz-1) +
// // 				  sdm_geom.solution_pts_1d_host(idz  ) ) * dz;
// // 		}
		
// // 		vertices.push_back(x);
// // 		vertices.push_back(y);
// // 		vertices.push_back(z);
		  
// // 	      } // for idx
// // 	    } // for idy
// // 	  } // for idz
	  
// // 	} // end for i
// //       } // end for j
// //     } // end for k

// //     uint64_t size = sizeof(float)*nx*ny*nz*(N+1)*(N+1)*(N+1)*3;
// //     outFile.write(reinterpret_cast<char *>( &size), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(vertices[0])), size);

// //     vertices.clear();
// //   }

// //   /*
// //    * Write connectivity.
// //    */
// //   {
// //     // this is only necessary for binary output
// //     std::vector<uint64_t> connectivity;

// //     // cell index
// //     uint64_t cell_index = 0;

// //     for (int k=0; k<nz; ++k) {
// //       for (int j=0; j<ny; ++j) {
// // 	for (int i=0; i<nx; ++i) {
	  
// // 	  uint64_t index = i+nx*j+nx*ny*k;
	  
// // 	  // offset to the first nodes in this cell
// // 	  uint64_t offset = index * nbNodesPerCell;
	  
// // 	  // loop over sub-cells
// // 	  for (int idz=0; idz<N; ++idz) {
// // 	    for (int idy=0; idy<N; ++idy) {
// // 	      for (int idx=0; idx<N; ++idx) {
		
// // 		connectivity.push_back(offset+idx  +N1* idy   + N2* idz   );
// // 		connectivity.push_back(offset+idx+1+N1* idy   + N2* idz   );
// // 		connectivity.push_back(offset+idx+1+N1*(idy+1)+ N2* idz   );
// // 		connectivity.push_back(offset+idx  +N1*(idy+1)+ N2* idz   );
// // 		connectivity.push_back(offset+idx  +N1* idy   + N2*(idz+1));
// // 		connectivity.push_back(offset+idx+1+N1* idy   + N2*(idz+1));
// // 		connectivity.push_back(offset+idx+1+N1*(idy+1)+ N2*(idz+1));
// // 		connectivity.push_back(offset+idx  +N1*(idy+1)+ N2*(idz+1));
		
// // 	      } // for idx
// // 	    } // for idy
// // 	  } // for idz
	  
// // 	  cell_index += nbSubCells;
	  
// // 	} // for i
// //       } // for j
// //     } // for k

// //     uint64_t size = sizeof(uint64_t)*nx*ny*nz*N*N*N*8;
// //     outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(connectivity[0]) ), size);
    
// //     connectivity.clear();
    
// //   } // end write connectivity
  
// //   /*
// //    * Write offsets.
// //    */
// //   {
// //     std::vector<uint64_t> offsets;

// //     // number of nodes per cell is 8 in 3D
// //     for (uint64_t i=1; i<=nx*ny*nz*N*N*N; ++i) {
// //       offsets.push_back(8*i);
// //     }

// //     uint64_t size = sizeof(uint64_t)*nx*ny*nz*N*N*N;
// //     outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(offsets[0]) ), size);
// //     offsets.clear();

// //   }

// //   /*
// //    * Write cell types.
// //    */
// //   {
// //     std::vector<unsigned char> celltypes;
    
// //     // 9 means "Quad" - 12 means "Hexahedron"
// //     for (uint64_t i=0; i<nx*ny*nz*N*N*N; ++i) {
// //       celltypes.push_back(12);
// //     }
    
// //     uint64_t size = sizeof(unsigned char)*nx*ny*nz*N*N*N;
// //     outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
// //     outFile.write(reinterpret_cast<char *>( &(celltypes[0]) ), size);
// //     celltypes.clear();
    
// //   }

// //   /*
// //    * Write cells data.
// //    */
// //   {
// //     const int gw = params.ghostWidth;
    
// //     std::vector<real_t> cells_data;

// //     // loop over scalar variables
// //     for ( int iVar=0; iVar<params.nbvar; iVar++) {

// //       // no ghost !!
// //       for (int k=0; k<nz; ++k) {
// // 	for (int j=0; j<ny; ++j) {
// // 	  for (int i=0; i<nx; ++i) {
	    
// // 	    // loop over sub-cells
// // 	    for (int idz=0; idz<N; ++idz) {
// // 	      for (int idy=0; idy<N; ++idy) {
// // 		for (int idx=0; idx<N; ++idx) {
		  
// // 		  real_t data = Uhost(gw+i,gw+j, gw+k, sdm::DofMap<3,N>(idx,idy,idz, iVar));

// // 		  cells_data.push_back(data);
		  
// // 		} // for idx
// // 	      } // for idy
// // 	    } // for idz
	  
// // 	  } // for i
// // 	} // for j
// //       } // for k

// //       uint64_t size = sizeof(real_t)*nx*ny*nz*N*N*N;
// //       outFile.write(reinterpret_cast<char *>( &size ), sizeof(uint64_t) );
// //       outFile.write(reinterpret_cast<char *>( &(cells_data[0]) ), size);
// //       cells_data.clear();
      
// //     } // end for iVar
    
// //   } // end write cells data

// //   outFile << " </AppendedData>" << "\n";

// // } // write_appended_binary_data - 3D

// ================================================================
// ================================================================
/**
 * 2D Output routine (VTK file format, ASCII, VtkUnstructuredGrid)
 * for High-Order Spectral Difference method schemes.

 * We use UnstructuredGrid here because, the mesh cells are unevenly
 * split into subcells, N x N+1 subcells
 *
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 *
 * Example usage in a SolverHydroSDM:
 *
 *   {
 *	DataArrayHost FluxesHost = Kokkos::create_mirror(Fluxes);
 *	ppkMHD::io::IO_Writer_SDM<dim,N>* p_io_writer = static_cast<typename ppkMHD::io::IO_Writer_SDM<dim,N>*>(m_io_writer.get());
 *	p_io_writer-> template save_flux<IX>(Fluxes, FluxesHost, m_times_saved, m_t, "debug1");
 *   }
 *
 *
 */
template<int N, int dir>
void save_VTK_SDM_Flux(DataArray2d             Udata,
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
  const int nx = params.nx;
  const int ny = params.ny;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;

  const int ghostWidth = params.ghostWidth;

  const int isize = params.isize;
  const int jsize = params.jsize;

  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  std::string dirStr;
  if (dir == IX)
    dirStr = "_Flux_x_";
  if (dir == IY)
    dirStr = "_Flux_y_";

  if ( !debug_name.empty() )
    dirStr += debug_name + "_";
  
  bool outputVtkAscii = true; //configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

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

#ifdef USE_MPI
  // write pvtu wrapper file
  if (params.myRank == 0) {
    
    // header file : parallel vtu format
    std::string headerFilename = outputDir+"/"+outputPrefix+dirStr+"_time"+stepNum.str()+".pvtu";
    
    write_pvtu_header(headerFilename,
		      outputPrefix,
		      params,
		      configMap,
		      nbvar,
		      variables_names,
		      iStep,
		      true);
  }
#endif // USE_MPI

  // concatenate file prefix + file number + suffix
  std::string filename;
  filename = outputDir + "/" + outputPrefix + dirStr + stepNum.str() + ".vtu";

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
  // each "cell" actually has a N x N+1 mini-mesh
  int nbSubCells = N*(N+1);
  int nbOfCells = nx*ny *  nbSubCells;

  int nbNodesPerCell = (N+1)*(N+2); // in 2D
  int nbOfNodes = nx*ny * nbNodesPerCell;
  
  outFile << "<Piece NumberOfPoints=\"" << nbOfNodes
	  <<"\" NumberOfCells=\"" << nbOfCells << "\" >\n";

  /*
   * write nodes location + data.
   */
  uint64_t offsetBytes = 0;

  write_nodes_location_flux<N,dir>(outFile,Uhost,sdm_geom,params,configMap,offsetBytes);

  write_cells_connectivity_flux<N,dir>(outFile, Uhost, sdm_geom, params, configMap,offsetBytes);

  write_flux_points_data<N,dir>(outFile, Uhost, sdm_geom, params, configMap, variables_names,offsetBytes);
  
  outFile << " </Piece>\n";
  
  outFile << " </UnstructuredGrid>\n";

  // write appended binary data (no compression, just raw binary)
  // UNIMPLEMENTED
  // if (!outputVtkAscii)
  //   write_appended_binary_data(outFile, Uhost, sdm_geom, params, configMap, variables_names);
  
  outFile << "</VTKFile>\n";
  
  outFile.close();
  
} // end save_VTK_SDM_Flux<N> - 2D

// ================================================================
// ================================================================
/**
 * 3D Output routine (VTK file format, ASCII, VtkUnstructuredGrid)
 * for High-Order Spectral Difference method schemes.

 * We use UnstructuredGrid here because, the mesh cells are unevenly
 * split into subcells, N x N+1 x N+1 subcells
 *
 * \param[in] Udata device data to save
 * \param[in,out] Uhost host data temporary array before saving to file
 */
template<int N,int dir>
void save_VTK_SDM_Flux(DataArray3d             Udata,
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
  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;

  const int kmin = params.kmin;
  const int kmax = params.kmax;

  const int ghostWidth = params.ghostWidth;

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;

  const int nbCells = isize*jsize*ksize;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,k,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  std::string dirStr;
  if (dir == IX)
    dirStr = "_Flux_x_";
  if (dir == IY)
    dirStr = "_Flux_y_";
  if (dir == IZ)
    dirStr = "_Flux_z_";

  if ( !debug_name.empty() )
    dirStr += debug_name + "_";

  bool outputVtkAscii = true; //configMap.getBool("output", "outputVtkAscii", false);
  const char *ascii_or_binary = outputVtkAscii ? "ascii" : "appended";

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
  
#ifdef USE_MPI
  // write pvtu wrapper file
  if (params.myRank == 0) {
    
    // header file : parallel vtu format
    std::string headerFilename = outputDir+"/"+outputPrefix+dirStr+"_time"+stepNum.str()+".pvtu";
    
    write_pvtu_header(headerFilename,
		      outputPrefix,
		      params,
		      configMap,
		      nbvar,
		      variables_names,
		      iStep,
		      true);
  }
#endif // USE_MPI

  // concatenate file prefix + file number + suffix
  std::string filename;
  filename = outputDir + "/" + outputPrefix + dirStr + stepNum.str() + ".vtu";
  
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
  // each "cell" actually has a N x N+1 x N+1 mini-mesh
  int nbSubCells = N*(N+1)*(N+1);
  int nbOfCells = nx*ny*nz *  nbSubCells;
  
  int nbNodesPerCell = (N+1)*(N+2)*(N+2); // in 3D
  int nbOfNodes = nx*ny*nz * nbNodesPerCell;
  
  outFile << "<Piece NumberOfPoints=\"" << nbOfNodes
	  <<"\" NumberOfCells=\"" << nbOfCells << "\" >\n";
  
  /*
   * write nodes location + data.
   */
  uint64_t offsetBytes = 0;
  
  write_nodes_location_flux<N,dir>(outFile,Uhost,sdm_geom,params,configMap,offsetBytes);
  
  write_cells_connectivity_flux<N,dir>(outFile, Uhost, sdm_geom, params, configMap,offsetBytes);
  
  write_flux_points_data<N,dir>(outFile, Uhost, sdm_geom, params, configMap, variables_names,offsetBytes);
  
  outFile << " </Piece>\n";
  
  outFile << " </UnstructuredGrid>\n";
  
  // write appended binary data (no compression, just raw binary)
  // UNIMPLEMENTED
  // if (!outputVtkAscii)
  //   write_appended_binary_data(outFile, Uhost, sdm_geom, params, configMap, variables_names);
  
  outFile << "</VTKFile>\n";

  outFile.close();
  
} // end save_VTK_SDM_Flux<N> - 3D

} // namespace io

} // namespace ppkMHD

#endif // IO_VTK_SDM_FLUX_H_
