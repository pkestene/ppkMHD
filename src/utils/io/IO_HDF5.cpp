#include <ctime>   // for std::time_t, std::tm, std::localtime

#include "IO_HDF5.h"

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
std::string current_date()
{
  
  /* get current time */
  std::time_t     now = std::time(nullptr);
  
  /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
  std::tm tm = *std::localtime(&now);
  
  // old versions of g++ don't have std::put_time,
  // so we provide a slight work arround
#if defined(__GNUC__) && (__GNUC__ < 5)
  
  char foo[64];
  
  std::strftime(foo, sizeof(foo), "%Y-%m-%d %H:%M:%S %Z", &tm);
  return std::string(foo);
  
#else
  
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %Z");
  
  return ss.str();

#endif

} // current_date

// =======================================================
// =======================================================
void writeXdmfForHdf5Wrapper(HydroParams& params,
			     ConfigMap& configMap,
			     int totalNumberOfSteps,
			     bool singleStep,
			     bool ghostIncluded)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const int ghostWidth = params.ghostWidth;

  const int dimType = params.dimType;

  const bool mhdEnabled = params.mhdEnabled;
  
  // data size actually written on disk
  int nxg = nx;
  int nyg = ny;
  int nzg = nz;
  if (ghostIncluded) {
    nxg += 2*ghostWidth;
    nyg += 2*ghostWidth;
    nzg += 2*ghostWidth;
  }

  // get data type as a string for Xdmf
  std::string dataTypeName;
  if (sizeof(real_t) == sizeof(float))
    dataTypeName = "Float";
  else
    dataTypeName = "Double";

  /*
   * 1. open XDMF and write header lines
   */
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
  std::string xdmfFilename = outputPrefix+".xmf";
  if (singleStep) { // add iStep to file name
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << totalNumberOfSteps;
    xdmfFilename = outputPrefix+"_"+outNum.str()+".xmf";
  }
  std::fstream xdmfFile;
  xdmfFile.open(xdmfFilename.c_str(), std::ios_base::out);

  xdmfFile << "<?xml version=\"1.0\" ?>"                       << std::endl;
  xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>"         << std::endl;
  xdmfFile << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
  xdmfFile << "  <Domain>"                                     << std::endl;
  xdmfFile << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

  // for each time step write a <grid> </grid> item
  int startStep=0;
  int stopStep =totalNumberOfSteps;
  int deltaStep=params.nOutput;
  if (deltaStep = -1)
    deltaStep=1;
  
  if (singleStep) {
    startStep = totalNumberOfSteps;
    stopStep  = totalNumberOfSteps+1;
    deltaStep = 1;
  }

  for (int iStep=startStep; iStep<=stopStep; iStep+=deltaStep) {
 
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << iStep;

    // take care that the following filename must be exactly the same as in routine outputHdf5 !!!
    std::string baseName         = outputPrefix+"_"+outNum.str();
    std::string hdf5Filename     = outputPrefix+"_"+outNum.str()+".h5";
    std::string hdf5FilenameFull = outputDir+"/"+outputPrefix+"_"+outNum.str()+".h5";

    xdmfFile << "    <Grid Name=\"" << baseName << "\" GridType=\"Uniform\">" << std::endl;
    xdmfFile << "    <Time Value=\"" << iStep << "\" />"                      << std::endl;
      
    // topology CoRectMesh
    if (dimType == TWO_D) 
      xdmfFile << "      <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\"" << nyg << " " << nxg << "\"/>" << std::endl;
    else
      xdmfFile << "      <Topology TopologyType=\"3DCoRectMesh\" NumberOfElements=\"" << nzg << " " << nyg << " " << nxg << "\"/>" << std::endl;
      
    // geometry
    if (dimType == TWO_D) {
      xdmfFile << "    <Geometry Type=\"ORIGIN_DXDY\">"        << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Origin\""                     << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"2\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       0 0"                                 << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Spacing\""                    << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"2\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       1 1"                                 << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    </Geometry>"                            << std::endl;
    } else {
      xdmfFile << "    <Geometry Type=\"ORIGIN_DXDYDZ\">"      << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Origin\""                     << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"3\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       0 0 0"                               << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Spacing\""                    << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"3\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       1 1 1"                               << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    </Geometry>"                            << std::endl;
    }
      
    // density
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"density\">" << std::endl;
    xdmfFile << "        <DataStructure"                             << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""    << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                         << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/density"             << std::endl;
    xdmfFile << "        </DataStructure>"                           << std::endl;
    xdmfFile << "      </Attribute>"                                 << std::endl;
      
    // energy
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"energy\">" << std::endl;
    xdmfFile << "        <DataStructure"                              << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""     << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                          << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/energy"             << std::endl;
    xdmfFile << "        </DataStructure>"                            << std::endl;
    xdmfFile << "      </Attribute>"                                  << std::endl;
      
    // momentum X
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_x\">" << std::endl;
    xdmfFile << "        <DataStructure"                                << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/momentum_x"             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
      
    // momentum Y
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_y\">" << std::endl;
    xdmfFile << "        <DataStructure" << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
    if (dimType == TWO_D)
      xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                            << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/momentum_y"             << std::endl;
    xdmfFile << "        </DataStructure>"                              << std::endl;
    xdmfFile << "      </Attribute>"                                    << std::endl;
      
    // momentum Z
    if (dimType == THREE_D and !mhdEnabled) {
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_z\">" << std::endl;
      xdmfFile << "        <DataStructure"                                << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/momentum_z"             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
    }
      
    if (mhdEnabled) {
      // momentum Z
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"momentum_z\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/momentum_z"             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;

      // magnetic field X
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_x\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_x"             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
	
      // magnetic field Y
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_y\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_y"             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
	
      // magnetic field Z
      xdmfFile << "      <Attribute Center=\"Node\" Name=\"magnetic_field_z\">" << std::endl;
      xdmfFile << "        <DataStructure" << std::endl;
      xdmfFile << "           DataType=\"" << dataTypeName <<  "\""       << std::endl;
      if (dimType == TWO_D)
	xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
      else
	xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
      xdmfFile << "           Format=\"HDF\">"                            << std::endl;
      xdmfFile << "           "<<hdf5Filename<<":/magnetic_field_z"             << std::endl;
      xdmfFile << "        </DataStructure>"                              << std::endl;
      xdmfFile << "      </Attribute>"                                    << std::endl;
	
    } // end mhdEnabled

      // finalize grid file for the current time step
    xdmfFile << "   </Grid>" << std::endl;
      
  } // end for loop over time step
    
    // finalize Xdmf wrapper file
  xdmfFile << "   </Grid>" << std::endl;
  xdmfFile << " </Domain>" << std::endl;
  xdmfFile << "</Xdmf>"    << std::endl;

} // writeXdmfForHdf5Wrapper

} // namespace io

} // namespace ppkMHD
