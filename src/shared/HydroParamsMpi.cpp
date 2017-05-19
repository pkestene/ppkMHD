#include "HydroParamsMpi.h"

using namespace hydroSimu;

// =======================================================
// =======================================================
void HydroParamsMpi::setup(ConfigMap& configMap)
{

  // call base class setup
  HydroParams::setup(configMap);

  // MPI parameters :
  mx = configMap.getInteger("mpi", "mx", 1);
  my = configMap.getInteger("mpi", "my", 1);
  mz = configMap.getInteger("mpi", "mz", 1);
    
  // check that parameters are consistent
  bool error = false;
  error |= (mx < 1);
  error |= (my < 1);
  error |= (mz < 1);

  // get world communicator size and check it is consistent with mesh grid sizes
  nProcs = MpiComm::world().getNProc();
  if (nProcs != mx*my*mz) {
    std::cerr << "Inconsistent MPI cartesian virtual topology geometry; \n mx*my*mz must match with parameter given to mpirun !!!\n";
    
  }

  // create the MPI communicator for our cartesian mesh
  if (dimType == TWO_D) {
    communicator = new MpiCommCart(mx, my, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
    nDim = 2;
  } else {
    communicator = new MpiCommCart(mx, my, mz, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
    nDim = 3;
  }
  
  // get my MPI rank inside topology
  myRank = communicator->getRank();
  
  // get my coordinates inside topology
  // myMpiPos[0] is between 0 and mx-1
  // myMpiPos[1] is between 0 and my-1
  // myMpiPos[2] is between 0 and mz-1
  myMpiPos.resize(nDim);
  communicator->getMyCoords(&myMpiPos[0]);
    
  /*
   * compute MPI ranks of our neighbors and 
   * set default boundary condition types
   */
  if (dimType == TWO_D) {
    nNeighbors = N_NEIGHBORS_2D;
    neighborsRank.resize(nNeighbors);
    neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
    neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
    neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
    neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
    
    neighborsBC.resize(nNeighbors);
    neighborsBC[X_MIN] = BC_COPY;
    neighborsBC[X_MAX] = BC_COPY;
    neighborsBC[Y_MIN] = BC_COPY;
    neighborsBC[Y_MAX] = BC_COPY;
  } else {
    nNeighbors = N_NEIGHBORS_3D;
    neighborsRank.resize(nNeighbors);
    neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
    neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
    neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
    neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
    neighborsRank[Z_MIN] = communicator->getNeighborRank<Z_MIN>();
    neighborsRank[Z_MAX] = communicator->getNeighborRank<Z_MAX>();
    
    neighborsBC.resize(nNeighbors);
    neighborsBC[X_MIN] = BC_COPY;
    neighborsBC[X_MAX] = BC_COPY;
    neighborsBC[Y_MIN] = BC_COPY;
    neighborsBC[Y_MAX] = BC_COPY;
    neighborsBC[Z_MIN] = BC_COPY;
    neighborsBC[Z_MAX] = BC_COPY;
  }
  
  /*
   * identify outside boundaries (no actual communication if we are
   * doing BC_DIRICHLET or BC_NEUMANN)
   *
   * Please notice the duality 
   * XMIN -- boundary_xmax
   * XMAX -- boundary_xmin
   *
   */
  
  // X_MIN boundary
  if (myMpiPos[DIR_X] == 0)
    neighborsBC[X_MIN] = boundary_type_xmin;
  
  // X_MAX boundary
  if (myMpiPos[DIR_X] == mx-1)
    neighborsBC[X_MAX] = boundary_type_xmax;
  
  // Y_MIN boundary
  if (myMpiPos[DIR_Y] == 0)
    neighborsBC[Y_MIN] = boundary_type_ymin;
  
  // Y_MAX boundary
  if (myMpiPos[DIR_Y] == my-1)
    neighborsBC[Y_MAX] = boundary_type_ymax;
  
  if (dimType == THREE_D) {
    
    // Z_MIN boundary
    if (myMpiPos[DIR_Z] == 0)
      neighborsBC[Z_MIN] = boundary_type_zmin;
    
    // Y_MAX boundary
    if (myMpiPos[DIR_Z] == mz-1)
      neighborsBC[Z_MAX] = boundary_type_zmax;
    
  } // end THREE_D
  
  /*
   * Initialize CUDA device if needed.
   * 
   * Let's assume hwloc is doing its job !
   *
   * Old comments from RamsesGPU:
   * When running on a Linux machine with mutiple GPU per node, it might be
   * very helpfull if admin has set the CUDA device compute mode to exclusive
   * so that a device is only attached to 1 host thread (i.e. 2 different host
   * thread can not communicate with the same GPU).
   *
   * As a sys-admin, just run for all devices command:
   *   nvidia-smi -g $(DEV_ID) -c 1
   *
   * If compute mode is set to normal mode, we need to use cudaSetDevice, 
   * so that each MPI device is mapped onto a different GPU device.
   * 
   * At CCRT, on machine Titane, each node (2 quadri-proc) "sees" only 
   * half a Tesla S1070, that means cudaGetDeviceCount should return 2.
   * If we want the ration 1 MPI process <-> 1 GPU, we need to allocate
   * N nodes and 2*N tasks (MPI process). 
   */
#ifdef CUDA
  // // get device count
  // int count;
  // cutilSafeCall( cudaGetDeviceCount(&count) );
  
  // int devId = myRank % count;
  // cutilSafeCall( cudaSetDevice(devId) );
  
  // cudaDeviceProp deviceProp;
  // int myDevId = -1;
  // cutilSafeCall( cudaGetDevice( &myDevId ) );
  // cutilSafeCall( cudaGetDeviceProperties( &deviceProp, myDevId ) );
  // // faire un cudaSetDevice et cudaGetDeviceProp et aficher le nom
  // // ajouter un booleen dans le constructeur pour savoir si on veut faire ca
  // // sachant que sur Titane, probablement que le mode exclusif est active
  // // a verifier demain
  
  // std::cout << "MPI process " << myRank << " is using GPU device num " << myDevId << std::endl;

#endif // CUDA

    // fix space resolution :
    // need to take into account number of MPI process in each direction
    dx = (xmax - xmin)/(nx*mx);
    dy = (ymax - ymin)/(ny*my);
    dz = (zmax - zmin)/(nz*mz);

    // print information about current setup
    if (myRank == 0) {
      std::cout << "We are about to start simulation with the following characteristics\n";

      std::cout << "Global resolution : " << 
	nx*mx << " x " << ny*my << " x " << nz*mz << "\n";
      std::cout << "Local  resolution : " << 
	nx << " x " << ny << " x " << nz << "\n";
      std::cout << "MPI Cartesian topology : " << mx << "x" << my << "x" << mz << std::endl;
    }
  
} // HydroParamsMpi::setup
