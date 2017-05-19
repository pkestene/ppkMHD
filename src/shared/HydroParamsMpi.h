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

struct HydroParamsMpi : HydroParams {
  
  using MpiCommCart = hydroSimu::MpiCommCart;
  
  //! size of the MPI cartesian grid
  int mx,my,mz;
  
  //! MPI communicator in a cartesian virtual topology
  MpiCommCart *communicator;
  
  //! number of dimension
  int nDim;
  
  //! MPI rank of current process
  int myRank;
  
  //! number of MPI processes
  int nProcs;
  
  //! MPI cartesian coordinates inside MPI topology
  std::vector<int> myMpiPos;
  
  //! number of MPI process neighbors (4 in 2D and 6 in 3D)
  int nNeighbors;
  
  //! MPI rank of adjacent MPI processes
  std::vector<int> neighborsRank;
  
  //! boundary condition type with adjacent domains (corresponding to
  //! neighbor MPI processes)
  std::vector<BoundaryConditionType> neighborsBC;
    
  //! constructor
  HydroParamsMpi() : HydroParams() {}

  void setup(ConfigMap& configMap);
  
}; // struct HydroParamsMpi


#endif // HYDRO_PARAMS_MPI_H_
