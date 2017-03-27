#include "shared/SolverFactory.h"

#include "shared/SolverBase.h"

#include "muscl/SolverHydroMuscl2D.h"
#include "muscl/SolverHydroMuscl3D.h"
#include "muscl/SolverMHDMuscl2D.h"
#include "muscl/SolverMHDMuscl3D.h"

#include "mood/SolverHydroMood.h"

namespace ppkMHD {

// The main solver creation routine
SolverFactory::SolverFactory()
{
  
  /*
   * Register some possible Solver/UserDataManager.
   */
  registerSolver("Hydro_Muscl_2D", &SolverHydroMuscl2D::create);
  registerSolver("Hydro_Muscl_3D", &SolverHydroMuscl3D::create);
  registerSolver("MHD_Muscl_2D",   &SolverMHDMuscl2D::create);
  registerSolver("MHD_Muscl_3D",   &SolverMHDMuscl3D::create);
  registerSolver("Hydro_Mood_2D_degree2",  &mood::SolverHydroMood<2,2>::create);
  	 
} // SolverFactory::SolverFactory

} // namespace ppkMHD
