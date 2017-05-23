#include <ctime>   // for std::time_t, std::tm, std::localtime

#include "IO_HDF5.h"

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
static std::string current_date()
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
			     bool mhdEnabled,
			     int totalNumberOfSteps,
			     bool singleStep,
			     bool ghostIncluded)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const int ghostWidth = params.ghostWidth;

  const int dimType = params.dimType;

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

// =======================================================
// =======================================================
void save_HDF5_2D(DataArray2d             Udata,
		  DataArray2d::HostMirror Uhost,
		  HydroParams& params,
		  ConfigMap& configMap,
		  bool mhdEnabled,
		  int nbvar,
		  const std::map<int, std::string>& variables_names,
		  int iStep,
		  real_t totalTime,
		  std::string debug_name)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;

  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;

  const int ghostWidth = params.ghostWidth;

  const int dimType = params.dimType;

  const bool ghostIncluded = configMap.getBool("output","ghostIncluded",false);

  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);

  herr_t status;
    
  // make filename string
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  std::ostringstream outNum;
  outNum.width(7);
  outNum.fill('0');
  outNum << iStep;
  std::string baseName         = outputPrefix+"_"+outNum.str();
  std::string hdf5Filename     = baseName+".h5";
  std::string hdf5FilenameFull = outputDir+"/"+hdf5Filename;
   
  // data size actually written on disk
  int nxg = nx;
  int nyg = ny;
  int nzg = nz;
  if (ghostIncluded) {
    nxg += 2*ghostWidth;
    nyg += 2*ghostWidth;
    nzg += 2*ghostWidth;
  }

  /*
   * write HDF5 file
   */
  // Create a new file using default properties.
  hid_t file_id = H5Fcreate(hdf5FilenameFull.c_str(), H5F_ACC_TRUNC |  H5F_ACC_DEBUG, H5P_DEFAULT, H5P_DEFAULT);

  // Create the data space for the dataset in memory and in file.
  hsize_t  dims_memory[3];
  hsize_t  dims_file[3];
  hid_t dataspace_memory, dataspace_file;
  if (dimType == TWO_D) {
    dims_memory[0] = jsize;
    dims_memory[1] = isize;
    dims_file[0] = nyg;
    dims_file[1] = nxg;
    dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
    dataspace_file   = H5Screate_simple(2, dims_file  , NULL);
  } else {
    dims_memory[0] = ksize;
    dims_memory[1] = jsize;
    dims_memory[2] = isize;
    dims_file[0] = nzg;
    dims_file[1] = nyg;
    dims_file[2] = nxg;
    dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
    dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
  }

  // Create the datasets.
  hid_t dataType;
  if (sizeof(real_t) == sizeof(float))
    dataType = H5T_NATIVE_FLOAT;
  else
    dataType = H5T_NATIVE_DOUBLE;
    

  // select data with or without ghost zones
  if (ghostIncluded) {
    if (dimType == TWO_D) {
      hsize_t  start[2] = {0, 0}; // ghost zone width
      hsize_t stride[2] = {1, 1};
      hsize_t  count[2] = {(hsize_t) nyg, (hsize_t) nxg};
      hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
      status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
    } else {
      hsize_t  start[3] = {0, 0, 0}; // ghost zone width
      hsize_t stride[3] = {1, 1, 1};
      hsize_t  count[3] = {(hsize_t) nzg, (hsize_t) nyg, (hsize_t) nxg};
      hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
      status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
    }      
  } else {
    if (dimType == TWO_D) {
      hsize_t  start[2] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
      hsize_t stride[2] = {1, 1};
      hsize_t  count[2] = {(hsize_t) ny, (hsize_t) nx};
      hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
      status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
    } else {
      hsize_t  start[3] = {(hsize_t) ghostWidth, (hsize_t) ghostWidth, (hsize_t) ghostWidth}; // ghost zone width
      hsize_t stride[3] = {1, 1, 1};
      hsize_t  count[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx};
      hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
      status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);      
    }
  }

  /*
   * property list for compression
   */
  // get compression level (0=no compression; 9 is highest level of compression)
  int compressionLevel = configMap.getInteger("output", "outputHdf5CompressionLevel", 0);
  if (compressionLevel < 0 or compressionLevel > 9) {
    std::cerr << "Invalid value for compression level; must be an integer between 0 and 9 !!!" << std::endl;
    std::cerr << "compression level is then set to default value 0; i.e. no compression !!" << std::endl;
    compressionLevel = 0;
  }

  hid_t propList_create_id = H5Pcreate(H5P_DATASET_CREATE);

  if (dimType == TWO_D) {
    const hsize_t chunk_size2D[2] = {(hsize_t) ny, (hsize_t) nx};
    H5Pset_chunk (propList_create_id, 2, chunk_size2D);
  } else { // THREE_D
    const hsize_t chunk_size3D[3] = {(hsize_t) nz, (hsize_t) ny, (hsize_t) nx};
    H5Pset_chunk (propList_create_id, 3, chunk_size3D);
  }
  H5Pset_shuffle (propList_create_id);
  H5Pset_deflate (propList_create_id, compressionLevel);
    
  /*
   * write heavy data to HDF5 file
   */
  real_t* data;

  // write density
  hid_t dataset_id = H5Dcreate2(file_id, "/density", dataType, dataspace_file, 
				H5P_DEFAULT, propList_create_id, H5P_DEFAULT);

  // Some adjustement needed to take into account that strides / layout need
  // to be checked at runtime
  
  // printf("KKKK %d %d %d\n",
  // 	 Uhost.dimension_0(),
  // 	 Uhost.dimension_1(),
  // 	 Uhost.dimension_2());
  // printf("KKKK %d %d %d\n",
  // 	 Uhost.stride_0(),
  // 	 Uhost.stride_1(),
  // 	 Uhost.stride_2());
  
  if (dimType == TWO_D)
    data = &(Uhost(0,0,ID));
  else
    data = &(Uhost(0,0,0,ID));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);

  // write total energy
  dataset_id = H5Dcreate2(file_id, "/energy", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  if (dimType == TWO_D)
    data = &(Uhost(0,0,IP));
  else
    data = &(Uhost(0,0,0,IP));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
  // write momentum X
  dataset_id = H5Dcreate2(file_id, "/momentum_x", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  if (dimType == TWO_D)
    data = &(Uhost(0,0,IU));
  else
    data = &(Uhost(0,0,0,IU));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
  // write momentum Y
  dataset_id = H5Dcreate2(file_id, "/momentum_y", dataType, dataspace_file, 
			  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
  if (dimType == TWO_D)
    data = &(Uhost(0,0,IV));
  else
    data = &(Uhost(0,0,0,IV));
  status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
  // write momentum Z (only if 3D hydro)
  if (dimType == THREE_D and !mhdEnabled) {
    dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    data = &(Uhost(0,0,0,IW));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
  }
    
  if (mhdEnabled) {
    // write momentum mz
    dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(Uhost(0,0,IW));
    else
      data = &(Uhost(0,0,0,IW));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
    // write magnetic field components
    dataset_id = H5Dcreate2(file_id, "/magnetic_field_x", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(Uhost(0,0,IA));
    else
      data = &(Uhost(0,0,0,IA));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
    dataset_id = H5Dcreate2(file_id, "/magnetic_field_y", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(Uhost(0,0,IB));
    else
      data = &(Uhost(0,0,0,IB));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
    dataset_id = H5Dcreate2(file_id, "/magnetic_field_z", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    if (dimType == TWO_D)
      data = &(Uhost(0,0,IC));
    else
      data = &(Uhost(0,0,0,IC));
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
  }

  // write time step as an attribute to root group
  hid_t ds_id;
  hid_t attr_id;
  {
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "time step", H5T_NATIVE_INT, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &iStep);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }
    
  // write total time 
  {
    double timeValue = (double) totalTime;

    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "total time", H5T_NATIVE_DOUBLE, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &timeValue);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }

  // write geometry information (just to be consistent)
  {
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "nx", H5T_NATIVE_INT, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &nx);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }
    
  {
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "ny", H5T_NATIVE_INT, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &ny);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }
    
  {
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "nz", H5T_NATIVE_INT, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &nz);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }

  // write information about ghost zone
  {
    int tmpVal = ghostIncluded ? 1 : 0;
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "ghost zone included", H5T_NATIVE_INT, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &tmpVal);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }

  // write date as an attribute to root group
  std::string dataString = current_date();
  const char *dataChar = dataString.c_str();
  hsize_t   dimsAttr[1] = {1};
  hid_t type = H5Tcopy (H5T_C_S1);
  status = H5Tset_size (type, H5T_VARIABLE);
  hid_t root_id = H5Gopen2(file_id, "/", H5P_DEFAULT);
  hid_t dataspace_id = H5Screate_simple(1, dimsAttr, NULL);
  attr_id = H5Acreate2(root_id, "creation date", type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attr_id, type, &dataChar);
  status = H5Aclose(attr_id);
  status = H5Gclose(root_id);
  status = H5Tclose(type);
  status = H5Sclose(dataspace_id);

  // close/release resources.
  H5Pclose(propList_create_id);
  H5Sclose(dataspace_memory);
  H5Sclose(dataspace_file);
  H5Dclose(dataset_id);
  H5Fflush(file_id, H5F_SCOPE_LOCAL);
  H5Fclose(file_id);

  (void) status;
  
} // save_HDF5_2D

} // namespace io

} // namespace ppkMHD
