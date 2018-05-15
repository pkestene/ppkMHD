#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "muscl/SolverHydroMuscl.h"
#include "shared/HydroParams.h"

#include "shared/mpiBorderUtils.h"

namespace ppkMHD { namespace muscl {

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<>
void SolverHydroMuscl<2>::make_boundaries(DataArray Udata)
{

  bool mhd_enabled = false;

#ifdef USE_MPI

  make_boundaries_mpi(Udata, mhd_enabled);

#else

  make_boundaries_serial(Udata, mhd_enabled);
  
#endif // USE_MPI
  
} // SolverHydroMuscl<2>::make_boundaries

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<>
void SolverHydroMuscl<3>::make_boundaries(DataArray Udata)
{

  bool mhd_enabled = false;

#ifdef USE_MPI

  make_boundaries_mpi(Udata, mhd_enabled);

#else

  make_boundaries_serial(Udata, mhd_enabled);
  
#endif // USE_MPI

} // SolverHydroMuscl<3>::make_boundaries

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::make_boundaries(DataArray Udata)
{

  // this routine is specialized for 2d / 3d
  
} // SolverHydroMuscl<dim>::make_boundaries


// =======================================================
// =======================================================
/**
 * Four quadrant 2D riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
template<>
void SolverHydroMuscl<2>::init_four_quadrant(DataArray Udata)
{

  int configNumber = configMap.getInteger("riemann2d","config_number",0);
  real_t xt = configMap.getFloat("riemann2d","x",0.8);
  real_t yt = configMap.getFloat("riemann2d","y",0.8);

  HydroState2d U0, U1, U2, U3;
  getRiemannConfig2d(configNumber, U0, U1, U2, U3);
  
  primToCons_2D(U0, params.settings.gamma0);
  primToCons_2D(U1, params.settings.gamma0);
  primToCons_2D(U2, params.settings.gamma0);
  primToCons_2D(U3, params.settings.gamma0);

  InitFourQuadrantFunctor2D::apply(params, Udata, configNumber,
				   U0, U1, U2, U3,
				   xt, yt, nbCells);
  
} // SolverHydroMuscl<2>::init_four_quadrant

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::init_four_quadrant(DataArray Udata)
{

  // specialized only for 2d
  std::cerr << "You shouldn't be here: four quadrant problem is not implemented in 3D !\n";
  
} // SolverHydroMuscl::init_four_quadrant

// =======================================================
// =======================================================
/**
 * Isentropic vortex advection test.
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 */
template<>
void SolverHydroMuscl<2>::init_isentropic_vortex(DataArray Udata)
{
  
  IsentropicVortexParams iparams(configMap);
  
  InitIsentropicVortexFunctor2D::apply(params, iparams, Udata, nbCells);
  
} // SolverHydroMuscl<2>::init_isentropic_vortex

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::init_isentropic_vortex(DataArray Udata)
{

  // specialized only for 2d
  std::cerr << "You shouldn't be here: isentropic vortex is not implemented in 3D !\n";
  
} // SolverHydroMuscl::init_isentropic_vortex

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::init_restart(DataArray Udata)
{

  // get input filename from configMap
  std::string inputFilename = configMap.getString("run", "restart_filename", "");
  
  // check filename extension
  std::string h5Suffix(".h5");
  std::string ncSuffix(".nc"); // pnetcdf file only available when MPI is activated
  
  bool isHdf5=false, isNcdf=false;
  if (inputFilename.length() >= 3) {
    isHdf5 = (0 == inputFilename.compare (inputFilename.length() -
					  h5Suffix.length(),
					  h5Suffix.length(),
					  h5Suffix) );
    isNcdf = (0 == inputFilename.compare (inputFilename.length() -
					  ncSuffix.length(),
					  ncSuffix.length(),
					  ncSuffix) );
  }

  // get output directory
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  
  // upscale init data from a file twice smaller
  bool restartUpscaleEnabled = configMap.getBool("run","restart_upscale",false);

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;
  const int ghostWidth = params.ghostWidth;
  const int nbvar = params.nbvar;

  int myRank=0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI
  
  if (restartUpscaleEnabled) { // load low resolution data from file
    
    // allocate h_input (half resolution, ghost included)
    DataArray input;
    if (dim == 2) 
      input = DataArray("h_input",
			nx/2+2*ghostWidth,
			ny/2+2*ghostWidth,
			nbvar);
    else if (dim==3)
      input = DataArray("h_input",
			nx/2+2*ghostWidth,
			ny/2+2*ghostWidth,
			nz/2+2*ghostWidth,
			nbvar);
    
    // read input date into temporary array h_input
    bool halfResolution=true;
    
    if (isHdf5) {
      //inputHdf5(input, outputDir+"/"+inputFilename, halfResolution);
    }
    
#ifdef USE_MPI
    else if (isNcdf) {
      //inputPnetcdf(h_input, outputDir+"/"+inputFilename, halfResolution);
    }
#endif // USE_MPI

    else {
      if (myRank == 0) {
	std::cerr << "Unknown input filename extension !\n";
	std::cerr << "Should be \".h5\" or \".nc\"\n";
      }
    }
    
    // upscale h_input into h_U (i.e. double resolution)
    //upscale(Udata, input);
    
  } else { // standard restart
    
    // read input file into h_U buffer , and return time Step
    if (isHdf5) {
      //timeStep = inputHdf5(h_U, outputDir+"/"+inputFilename);
    }
    
#ifdef USE_MPI
    else if (isNcdf) {
      //timeStep = inputPnetcdf(h_U, outputDir+"/"+inputFilename);
    }
#endif // USE_MPI
    
    else {
      if (myRank == 0) {
	std::cerr << "Unknown input filename extension !\n";
	std::cerr << "Should be \".h5\" or \".nc\"\n";
      }
    }
    
  } // if (restartUpscaleEnabled)
  
  // in case of turbulence problem, we also need to re-initialize the
  // random forcing field
  // if (!problemName.compare("turbulence")) {
  //   this->init_randomForcing();
  // } 
  
  // in case of Ornstein-Uhlenbeck turbulence problem, 
  // we also need to re-initialize the random forcing field
  // if (!problemName.compare("turbulence-Ornstein-Uhlenbeck")) {
    
  //   bool restartEnabled = true;
    
  //   std::string forcing_filename = configMap.getString("turbulence-Ornstein-Uhlenbeck", "forcing_input_file",  "");
    
  //   if (restartUpscaleEnabled) {
      
  //     // use default parameter when restarting and upscaling
  //     pForcingOrnsteinUhlenbeck -> init_forcing(false);
      
  //   } else if ( forcing_filename.size() != 0) {
      
  //     // if forcing filename is provided, we use it
  //     pForcingOrnsteinUhlenbeck -> init_forcing(false); // call to allocate
  //     pForcingOrnsteinUhlenbeck -> input_forcing(forcing_filename);
      
  //   } else {
      
  //     // the forcing parameter filename is build upon configMap information
  //     pForcingOrnsteinUhlenbeck -> init_forcing(restartEnabled, timeStep);
      
  //   }
    
  // } // end restart problem turbulence-Ornstein-Uhlenbeck
  
  // some extra stuff that need to be done here (usefull when MRI is activated)
  //restart_run_extra_work();

  
} // SolverHydroMuscl<2>::init_restart

// =======================================================
// =======================================================
template<>
void SolverHydroMuscl<2>::init(DataArray Udata)
{

  // test if we are performing a re-start run (default : false)
  bool restartEnabled = configMap.getBool("run","restart",false);

  if (restartEnabled) { // load data from input data file

    init_restart(Udata);
    
  } else { // regular initialization

    /*
     * initialize hydro array at t=0
     */
    if ( !m_problem_name.compare("implode") ) {
      
      init_implode(Udata);
      
    } else if ( !m_problem_name.compare("blast") ) {
      
      init_blast(Udata);
      
    } else if ( !m_problem_name.compare("four_quadrant") ) {
      
      init_four_quadrant(Udata);
      
    } else if ( !m_problem_name.compare("isentropic_vortex") ) {
      
      init_isentropic_vortex(Udata);
      
    } else if ( !m_problem_name.compare("rayleigh_taylor") ) {
      
      init_rayleigh_taylor(Udata,gravity);
      
    } else {
      
      std::cout << "Problem : " << m_problem_name
		<< " is not recognized / implemented."
		<< std::endl;
      std::cout <<  "Use default - implode" << std::endl;
      init_implode(Udata);
      
    }
    
  } // end regular initialization

} // SolverHydroMuscl::init / 2d

// =======================================================
// =======================================================
template<>
void SolverHydroMuscl<3>::init(DataArray Udata)
{

  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(Udata);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(Udata);

  } else if ( !m_problem_name.compare("rayleigh_taylor") ) {
    
    init_rayleigh_taylor(Udata,gravity);
    
  } else {
    
    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - implode" << std::endl;
    init_implode(Udata);

  }

} // SolverHydroMuscl<3>::init

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 2d
// ///////////////////////////////////////////
template<>
void SolverHydroMuscl<2>::godunov_unsplit_impl(DataArray data_in, 
					       DataArray data_out, 
					       real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

  // fill ghost cell in data_in
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  timers[TIMER_BOUNDARIES]->stop();
    
  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  if (params.implementationVersion == 0) {
    
    // compute fluxes (if gravity_enabled is false, the last parameter is not used)
    ComputeAndStoreFluxesFunctor2D::apply(params, Q,
					  Fluxes_x, Fluxes_y,
					  dt,
					  m_gravity_enabled,
					  gravity);
    
    // actual update
    UpdateFunctor2D::apply(params, data_out,
			   Fluxes_x, Fluxes_y,
			   nbCells);

    // gravity source term
    if (m_gravity_enabled) {
      GravitySourceTermFunctor2D::apply(params, data_in, data_out, gravity, dt);
    }

    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor2D::apply(params, Q,
				  Slopes_x, Slopes_y, nbCells);

    // now trace along X axis
    ComputeTraceAndFluxes_Functor2D<XDIR>::apply(params, Q,
						 Slopes_x, Slopes_y,
						 Fluxes_x,
						 dtdx, dtdy, nbCells);
    
    // and update along X axis
    UpdateDirFunctor2D<XDIR>::apply(params, data_out, Fluxes_x, nbCells);
    
    // now trace along Y axis
    ComputeTraceAndFluxes_Functor2D<YDIR>::apply(params, Q,
						 Slopes_x, Slopes_y,
						 Fluxes_y,
						 dtdx, dtdy, nbCells);
    
    // and update along Y axis
    UpdateDirFunctor2D<YDIR>::apply(params, data_out, Fluxes_y, nbCells);
    
  } // end params.implementationVersion == 1
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMuscl2D::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 3d
// ///////////////////////////////////////////
template<>
void SolverHydroMuscl<3>::godunov_unsplit_impl(DataArray data_in, 
					       DataArray data_out, 
					       real_t dt)
{
  real_t dtdx;
  real_t dtdy;
  real_t dtdz;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  // fill ghost cell in data_in
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  timers[TIMER_BOUNDARIES]->stop();
    
  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  if (params.implementationVersion == 0) {
    
    // compute fluxes
    ComputeAndStoreFluxesFunctor3D::apply(params, Q,
					  Fluxes_x, Fluxes_y, Fluxes_z,
					  dtdx, dtdy, dtdz,
					  nbCells);

    // actual update
    UpdateFunctor3D::apply(params, data_out,
			   Fluxes_x, Fluxes_y, Fluxes_z,
			   nbCells);
    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor3D::apply(params, Q,
				  Slopes_x, Slopes_y, Slopes_z,
				  nbCells);

    // now trace along X axis
    ComputeTraceAndFluxes_Functor3D<XDIR>::apply(params, Q,
						 Slopes_x, Slopes_y, Slopes_z,
						 Fluxes_x,
						 dtdx, dtdy, dtdz, nbCells);
    
    // and update along X axis
    UpdateDirFunctor3D<XDIR>::apply(params, data_out, Fluxes_x, nbCells);

    // now trace along Y axis
    ComputeTraceAndFluxes_Functor3D<YDIR>::apply(params, Q,
						 Slopes_x, Slopes_y, Slopes_z,
						 Fluxes_y,
						 dtdx, dtdy, dtdz, nbCells);
    
    // and update along Y axis
    UpdateDirFunctor3D<YDIR>::apply(params, data_out, Fluxes_y, nbCells);

    // now trace along Z axis
    ComputeTraceAndFluxes_Functor3D<ZDIR>::apply(params, Q,
						 Slopes_x, Slopes_y, Slopes_z,
						 Fluxes_z,
						 dtdx, dtdy, dtdz, nbCells);
    
    // and update along Z axis
    UpdateDirFunctor3D<ZDIR>::apply(params, data_out, Fluxes_z, nbCells);

  } // end params.implementationVersion == 1
  
  timers[TIMER_NUM_SCHEME]->stop();

} // SolverHydroMuscl<3>::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
template<int dim>
void SolverHydroMuscl<dim>::godunov_unsplit_impl(DataArray data_in, 
						 DataArray data_out, 
						 real_t dt)
{

  // 2d / 3d implementation are specialized in implementation file
  
} // SolverHydroMuscl3D::godunov_unsplit_impl

} // namespace muscl

} // namespace ppkMHD
