#include "SolverFactory.h"

#include "SolverBase.h"

#include "SolverHydroMuscl2D.h"
#include "SolverHydroMuscl3D.h"
#include "SolverMHDMuscl2D.h"
#include "SolverMHDMuscl3D.h"

namespace ppkMHD {

// The main solver creation routine
SolverFactory::SolverFactory()
{
  
  /*
   * Register some possible Solver/UserDataManager.
   */
  registerSolver("Hydro_Muscl_2D", &SolverHydroMuscl2D::create);
  registerSolver("Hydro_Muscl_3D", &SolverHydroMuscl3D::create);
  //registerSolver("MHD_Muscl_2D",   &SolverMHDMuscl2D::create);
  //registerSolver("MHD_Muscl_3D",   &SolverMHDMuscl3D::create);
  	 
} // SolverFactory::SolverFactory

} // namespace ppkMHD
