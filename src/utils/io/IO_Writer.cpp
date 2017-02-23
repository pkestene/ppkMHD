#include "IO_Writer.h"

// =======================================================
// =======================================================
IO_Writer::IO_Writer(HydroParams& params, ConfigMap& configMap) :
  params(params),
  configMap(configMap),
  vtkEnabled(true),
  hdf5Enabled(false),
  pnetcdfEnabled(false)
{

} // IO_Writer
