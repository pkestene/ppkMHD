#ifndef IO_HDF5_H_
#define IO_HDF5_H_

#include <iostream>
#include <type_traits>

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
//class HydroParams;
//class ConfigMap;
#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"

namespace ppkMHD { namespace io {

// =======================================================
// =======================================================
/**
 * Return current date in a string.
 */
std::string current_date();

// =======================================================
// =======================================================
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
			     int totalNumberOfSteps,
			     bool singleStep);

// =======================================================
// =======================================================
/**
 *
 */
template<DimensionType d>
class Save_HDF5
{
public:
  //! Decide at compile-time which data array type to use
  using DataArray  = typename std::conditional<d==TWO_D,DataArray2d,DataArray3d>::type;
  using DataArrayHost  = typename std::conditional<d==TWO_D,DataArray2dHost,DataArray3dHost>::type;

  Save_HDF5(DataArray     Udata,
	    DataArrayHost Uhost,
	    HydroParams& params,
	    ConfigMap& configMap,
	    int nbvar,
	    const std::map<int, std::string>& variables_names,
	    int iStep,
	    real_t totalTime,
	    std::string debug_name) :
    Udata(Udata), Uhost(Uhost), params(params), configMap(configMap),
    nbvar(nbvar), variables_names(variables_names),
    iStep(iStep), totalTime(totalTime), debug_name(debug_name)
  {};
  ~Save_HDF5() {};

  template<DimensionType d_ = d>
  void copy_buffer(typename std::enable_if<d_==TWO_D, real_t>::type *data,
		   int isize, int jsize, int ksize, int nvar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
      for (int j=0; j<jsize; ++j) {
	for (int i=0; i<isize; ++i) {
	  int index = i+isize*j;
	  data[index]=Uhost(i,j,nvar);
	}
      }
    } else {
      data = &(Uhost(0,0,nvar));
    }

  } // copy_buffer

  template<DimensionType d_=d>
  void copy_buffer(typename std::enable_if<d_==THREE_D, real_t>::type *data,
		   int isize, int jsize, int ksize, int nvar, KokkosLayout layout)
  {
    if (layout == KOKKOS_LAYOUT_RIGHT) { // transpose array to make data contiguous in memory
      for (int k=0; k<ksize; ++k) {
	for (int j=0; j<jsize; ++j) {
	  for (int i=0; i<isize; ++i) {
	    int index = i+isize*j+isize*jsize*k;
	    data[index]=Uhost(i,j,k,nvar);
	  }
	}
      }
      
    } else {
      data = &(Uhost(0,0,0,nvar));
    }

  } // copy_buffer / 3D
  
  // =======================================================
  // =======================================================
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
  void save()
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

    const bool mhdEnabled = params.mhdEnabled;
    
    // copy device data to host
    Kokkos::deep_copy(Uhost, Udata);

    // here we need to check Uhost memory layout
    KokkosLayout layout;
    if (Uhost.stride_0()==1 and Uhost.stride_0()!=1)
      layout = KOKKOS_LAYOUT_LEFT;
    if (Uhost.stride_0()>1)
      layout = KOKKOS_LAYOUT_RIGHT;
  
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
  
    // Some adjustement needed to take into account that strides / layout need
    // to be checked at runtime
    // if memory layout is KOKKOS_LAYOUT_RIGHT, we need an extra buffer.
    if (layout == KOKKOS_LAYOUT_RIGHT) {

      if (dimType == TWO_D)
	data = new real_t[isize*jsize];
      else
	data = new real_t[isize*jsize*ksize];

    }
  
    // write density
    hid_t dataset_id = H5Dcreate2(file_id, "/density", dataType, dataspace_file, 
				  H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, ID, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
  
    // write total energy
    dataset_id = H5Dcreate2(file_id, "/energy", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IP, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write momentum X
    dataset_id = H5Dcreate2(file_id, "/momentum_x", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IU, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
    // write momentum Y
    dataset_id = H5Dcreate2(file_id, "/momentum_y", dataType, dataspace_file, 
			    H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
    copy_buffer(data, isize, jsize, ksize, IV, layout);
    status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
  
    // write momentum Z (only if 3D hydro)
    if (dimType == THREE_D and !mhdEnabled) {
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IW, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    }
    
    if (mhdEnabled) {
      // write momentum mz
      dataset_id = H5Dcreate2(file_id, "/momentum_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IW, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
      // write magnetic field components
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_x", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IA, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_y", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IB, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
    
      dataset_id = H5Dcreate2(file_id, "/magnetic_field_z", dataType, dataspace_file, 
			      H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
      copy_buffer(data, isize, jsize, ksize, IC, layout);
      status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);
     
    }

    // free memory if necessary
    if (layout == KOKKOS_LAYOUT_RIGHT) {
      delete[] data;
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
  
  } // save

  
  DataArray     Udata;
  DataArrayHost Uhost;
  HydroParams& params;
  ConfigMap& configMap;
  int nbvar;
  const std::map<int, std::string>& variables_names;
  int iStep;
  real_t totalTime;
  std::string debug_name;
  
}; // class Save_HDF5


} // namespace io

} // namespace ppkMHD

#endif // IO_HDF5_H_
