#ifndef IO_WRITER_H_
#define IO_WRITER_H_

#include <HydroParams.h>
#include <config/ConfigMap.h>
#include <kokkos_shared.h>

namespace ppkMHD { namespace io {

/**
 * 
 */
class IO_Writer {

public:
  IO_Writer(HydroParams& params, ConfigMap& configMap);
  virtual ~IO_Writer();

  HydroParams& params;
  ConfigMap& configMap;

  template<DimensionType dimType>
  void save_data(DataArray);

  bool vtkEnabled;
  bool vtkHdf5Enabled;
  bool pnetcdfEnabled;
  
  // private:
  //   void save_vtk(DataArray);
  //   void save_hdf5(DataArray);
  //   void save_pnetcdf(DataArray);
  
}; // class IO_Writer


} // namespace io

} // namespace ppkMHD

#endif // IO_WRITER_H_
