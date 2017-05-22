#include "shared/SolverFactory.h"

#include "shared/SolverBase.h"

#include "muscl/SolverHydroMuscl2D.h"
#include "muscl/SolverHydroMuscl3D.h"
#include "muscl/SolverMHDMuscl2D.h"
#include "muscl/SolverMHDMuscl3D.h"

#ifdef USE_MOOD
#include "mood/SolverHydroMood.h"
#endif // USE_MOOD

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

#ifdef USE_MOOD
  registerSolver("Hydro_Mood_2D_degree1",  &mood::SolverHydroMood<2,1>::create);
  registerSolver("Hydro_Mood_2D_degree2",  &mood::SolverHydroMood<2,2>::create);
  registerSolver("Hydro_Mood_2D_degree3",  &mood::SolverHydroMood<2,3>::create);
  registerSolver("Hydro_Mood_2D_degree4",  &mood::SolverHydroMood<2,4>::create);
  registerSolver("Hydro_Mood_3D_degree1",  &mood::SolverHydroMood<3,1>::create);
#endif // USE_MOOD
  
} // SolverFactory::SolverFactory

} // namespace ppkMHD
