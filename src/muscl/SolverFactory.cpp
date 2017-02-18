#include "SolverFactory.h"

#include "SolverBase.h"

#include "SolverMuscl2D.h"
#include "SolverMuscl3D.h"


// The main solver creation routine
SolverFactory::SolverFactory()
{
  
  /*
   * Register some possible Solver/UserDataManager.
   */
  registerSolver("HydroMuscl2D", &SolverMuscl2D::create);
  registerSolver("HydroMuscl3D", &SolverMuscl3D::create);
  	 
} // SolverFactory::SolverFactory
