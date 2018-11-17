#include "shared/SolverFactory.h"

#include "shared/SolverBase.h"

#include "muscl/SolverHydroMuscl.h"
#include "muscl/SolverMHDMuscl.h"

#ifdef USE_SDM
#include "sdm/SolverHydroSDM.h"
#endif // USE_SDM

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
  registerSolver("Hydro_Muscl_2D", &muscl::SolverHydroMuscl<2>::create);
  registerSolver("Hydro_Muscl_3D", &muscl::SolverHydroMuscl<3>::create);

  registerSolver("MHD_Muscl_2D",   &muscl::SolverMHDMuscl<2>::create);
  registerSolver("MHD_Muscl_3D",   &muscl::SolverMHDMuscl<3>::create);

#ifdef USE_SDM
  registerSolver("Hydro_SDM_2D_degree1",   &sdm::SolverHydroSDM<2,1>::create);
  registerSolver("Hydro_SDM_2D_degree2",   &sdm::SolverHydroSDM<2,2>::create);
  registerSolver("Hydro_SDM_2D_degree3",   &sdm::SolverHydroSDM<2,3>::create);
  registerSolver("Hydro_SDM_2D_degree4",   &sdm::SolverHydroSDM<2,4>::create);
  registerSolver("Hydro_SDM_2D_degree5",   &sdm::SolverHydroSDM<2,5>::create);
  registerSolver("Hydro_SDM_2D_degree6",   &sdm::SolverHydroSDM<2,6>::create);
  
  registerSolver("Hydro_SDM_3D_degree2",   &sdm::SolverHydroSDM<3,2>::create);
  registerSolver("Hydro_SDM_3D_degree3",   &sdm::SolverHydroSDM<3,3>::create);
  registerSolver("Hydro_SDM_3D_degree4",   &sdm::SolverHydroSDM<3,4>::create);
#endif // USE_SDM
  
#ifdef USE_MOOD
  registerSolver("Hydro_Mood_2D_degree1",  &mood::SolverHydroMood<2,1>::create);
  registerSolver("Hydro_Mood_2D_degree2",  &mood::SolverHydroMood<2,2>::create);
  registerSolver("Hydro_Mood_2D_degree3",  &mood::SolverHydroMood<2,3>::create);
  registerSolver("Hydro_Mood_2D_degree4",  &mood::SolverHydroMood<2,4>::create);
  registerSolver("Hydro_Mood_3D_degree1",  &mood::SolverHydroMood<3,1>::create);
#endif // USE_MOOD
  
} // SolverFactory::SolverFactory

} // namespace ppkMHD
