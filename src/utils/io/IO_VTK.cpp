#include "IO_VTK.h"

#include <shared/HydroParams.h>
#include <utils/config/ConfigMap.h>

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
static bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}

// =======================================================
// =======================================================
void save_VTK_2D(DataArray2d             Udata,
		 DataArray2d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep,
		 std::string debug_name)
{
  const int nx = params.nx;
  const int ny = params.ny;

  const int imin = params.imin;
  const int imax = params.imax;

  const int jmin = params.jmin;
  const int jmax = params.jmax;

  const int ghostWidth = params.ghostWidth;

  const int isize = params.isize;
  const int ijsize = params.isize * params.jsize;
  
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
  if ( debug_name.empty() )
    filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vti";
  else
    filename = outputDir + "/" + outputPrefix + "_" + debug_name + "_" + stepNum.str() + ".vti";
  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header

  // if writing raw binary data (file does not respect XML standard)
  if (outputVtkAscii)
    outFile << "<?xml version=\"1.0\"?>\n";

  // write xml data header
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 0  << " "
	  <<  "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 1  << " "    
	  << "\">\n";
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";

  if (outputVtkAscii) {

    outFile << "    <CellData>\n";

    // write data array (ascii), remove ghost cells
    for ( iVar=0; iVar<nbvar; iVar++) {
      outFile << "    <DataArray type=\"";
      if (useDouble)
	outFile << "Float64";
      else
	outFile << "Float32";
      outFile << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"ascii\" >\n";
      
      for (int index=0; index<ijsize; ++index) {
	//index2coord(index,i,j,isize,jsize);
	
	// enforce the use of left layout (Ok for CUDA)
	// but for OpenMP, we will need to transpose
	j = index / isize;
	i = index - j*isize;
	
	if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	    i>=imin+ghostWidth and i<=imax-ghostWidth) {
	  outFile << Uhost(i, j, iVar) << " ";
	}
      }
      outFile << "\n    </DataArray>\n";
    } // end for iVar

    outFile << "    </CellData>\n";
    
    // write footer
    outFile << "  </Piece>\n";
    outFile << "  </ImageData>\n";
    outFile << "</VTKFile>\n";

  } else { // write data in binary format

    outFile << "    <CellData>" << std::endl;

    for (int iVar=0; iVar<nbvar; iVar++) {
      if (useDouble) {
	outFile << "     <DataArray type=\"Float64\" Name=\"" ;
      } else {
	outFile << "     <DataArray type=\"Float32\" Name=\"" ;
      }
      outFile << variables_names.at(iVar)
	      << "\" format=\"appended\" offset=\""
	      << iVar*nx*ny*sizeof(real_t)+iVar*sizeof(unsigned int)
	      <<"\" />" << std::endl;
    }

    outFile << "    </CellData>" << std::endl;
    outFile << "  </Piece>" << std::endl;
    outFile << "  </ImageData>" << std::endl;
    
    outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

    // write the leading undescore
    outFile << "_";
    // then write heavy data (column major format)
    {
      unsigned int nbOfWords = nx*ny*sizeof(real_t);
      for (int iVar=0; iVar<nbvar; iVar++) {
	outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	for (int j=jmin+ghostWidth; j<=jmax-ghostWidth; j++)
	  for (int i=imin+ghostWidth; i<=imax-ghostWidth; i++) {
	    real_t tmp = Uhost(i, j, iVar);
	    outFile.write((char *)&tmp,sizeof(real_t));
	  }
      }
    }

    outFile << "  </AppendedData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;

  } // end ascii/binary heavy data write

  
  outFile.close();
  
} // end save_VTK_2D

// =======================================================
// =======================================================
void save_VTK_3D(DataArray3d             Udata,
		 DataArray3d::HostMirror Uhost,
		 HydroParams& params,
		 ConfigMap& configMap,
		 int nbvar,
		 const std::map<int, std::string>& variables_names,
		 int iStep,
		 std::string debug_name)
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

  const int isize = params.isize;
  const int ijsize = params.isize * params.jsize;
  const int ijksize = params.isize * params.jsize * params.ksize;

  
  const int ghostWidth = params.ghostWidth;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i, j, k, iVar;
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
  if ( debug_name.empty() )
    filename = outputDir + "/" + outputPrefix + "_" + stepNum.str() + ".vti";
  else
    filename = outputDir + "/" + outputPrefix + "_" + debug_name + "_" + stepNum.str() + ".vti";
  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header

  // if writing raw binary data (file does not respect XML standard)
  if (outputVtkAscii)
    outFile << "<?xml version=\"1.0\"?>\n";
  
  // write xml data header
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nz  << " "
	  <<  "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nz << " "    
	  << "\">\n";
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";

  if (outputVtkAscii) {
    
    outFile << "    <CellData>\n";
    
    // write data array (ascii), remove ghost cells
    for ( iVar=0; iVar<nbvar; iVar++) {
      outFile << "    <DataArray type=\"";
      if (useDouble)
	outFile << "Float64";
      else
	outFile << "Float32";
      outFile << "\" Name=\"" << variables_names.at(iVar) << "\" format=\"ascii\" >\n";
      
      for (int index=0; index<ijksize; ++index) {
	//index2coord(index,i,j,k,isize,jsize,ksize);
	
	// enforce the use of left layout (Ok for CUDA)
	// but for OpenMP, we will need to transpose
	k = index / ijsize;
	j = (index - k*ijsize) / isize;
	i = index - j*isize - k*ijsize;
	
	if (k>=kmin+ghostWidth and k<=kmax-ghostWidth and
	    j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	    i>=imin+ghostWidth and i<=imax-ghostWidth) {
    	outFile << Uhost(i,j,k,iVar) << " ";
	}
      }
      outFile << "\n    </DataArray>\n";
    } // end for iVar

    outFile << "    </CellData>\n";

    // write footer
    outFile << "  </Piece>\n";
    outFile << "  </ImageData>\n";
    outFile << "</VTKFile>\n";

  } else { // write data in binary format

    outFile << "    <CellData>" << std::endl;

    for (int iVar=0; iVar<nbvar; iVar++) {
      if (useDouble) {
	outFile << "     <DataArray type=\"Float64\" Name=\"" ;
      } else {
	outFile << "     <DataArray type=\"Float32\" Name=\"" ;
      }
      outFile << variables_names.at(iVar)
	      << "\" format=\"appended\" offset=\""
	      << iVar*nx*ny*nz*sizeof(real_t)+iVar*sizeof(unsigned int)
	      <<"\" />" << std::endl;
    }

    outFile << "    </CellData>" << std::endl;
    outFile << "  </Piece>" << std::endl;
    outFile << "  </ImageData>" << std::endl;
    
    outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

    // write the leading undescore
    outFile << "_";

    // then write heavy data (column major format)
    {
      unsigned int nbOfWords = nx*ny*nz*sizeof(real_t);
      for (int iVar=0; iVar<nbvar; iVar++) {
	outFile.write((char *)&nbOfWords,sizeof(unsigned int));
	 for (int k=kmin+ghostWidth; k<=kmax-ghostWidth; k++)
	   for (int j=jmin+ghostWidth; j<=jmax-ghostWidth; j++)
	     for (int i=imin+ghostWidth; i<=imax-ghostWidth; i++) {
	       real_t tmp = Uhost(i, j, k, iVar);
	       outFile.write((char *)&tmp,sizeof(real_t));
	     }
      }
    }

    outFile << "  </AppendedData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;

  } // end ascii/binary heavy data write
  
  outFile.close();

} // end save_VTK_3D

} // namespace io

} // namespace ppkMHD
