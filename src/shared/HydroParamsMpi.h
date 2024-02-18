/**
 * \file HydroParamsMpi.h
 * \brief Hydrodynamics solver parameters for an MPI run.
 *
 * \date May, 19 2017
 * \author P. Kestener
 */
#ifndef HYDRO_PARAMS_MPI_H_
#define HYDRO_PARAMS_MPI_H_

#include <vector>

#include "HydroParams.h"

#include "utils/mpiUtils/MpiCommCart.h"

namespace ppkMHD {

struct HydroParamsMpi : HydroParams
{

  //! constructor
  HydroParamsMpi()
    : HydroParams()
  {}

  void setup(ConfigMap& configMap);

}; // struct HydroParamsMpi

} // namespace ppkMHD

#endif // HYDRO_PARAMS_MPI_H_
