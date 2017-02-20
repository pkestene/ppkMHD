#include "SolverFactory.h"

#include "SolverBase.h"

#include "SolverMuscl2D.h"
#include "SolverMuscl3D.h"
#include "SolverMHDMuscl2D.h"

namespace ppkMHD {

// The main solver creation routine
SolverFactory::SolverFactory()
{
  
  /*
   * Register some possible Solver/UserDataManager.
   */
  registerSolver("Hydro_Muscl_2D", &SolverMuscl2D::create);
  registerSolver("Hydro_Muscl_3D", &SolverMuscl3D::create);
  registerSolver("MHD_Muscl_2D",   &SolverMHDMuscl2D::create);
  	 
} // SolverFactory::SolverFactory

} // namespace ppkMHD
