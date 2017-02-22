#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "SolverHydroMuscl2D.h"
#include "HydroParams.h"

// the actual computational functors called in HydroRun
#include "HydroRunFunctors2D.h"

// Kokkos
#include "kokkos_shared.h"

// for init condition
#include "BlastParams.h"

static bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}

namespace ppkMHD {

using namespace muscl::hydro2d;

// =======================================================
// ==== CLASS SolverHydroMuscl2D IMPL ====================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl2D::SolverHydroMuscl2D(HydroParams& params, ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(),
  Slopes_x(), Slopes_y(),
  isize(params.isize),
  jsize(params.jsize),
  ijsize(params.isize*params.jsize)
{

  m_nCells = ijsize;

  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", isize, jsize, nbvar);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",isize, jsize, nbvar);
  Q     = DataArray("Q", isize, jsize, nbvar);

  if (params.implementationVersion == 0) {

    Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);
    
  } else if (params.implementationVersion == 1) {

    Slopes_x = DataArray("Slope_x", isize, jsize, nbvar);
    Slopes_y = DataArray("Slope_y", isize, jsize, nbvar);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
    Fluxes_y = Fluxes_x;
    
  } 
  
  // default riemann solver
  // riemann_solver_fn = &SolverHydroMuscl2D::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &SolverHydroMuscl2D::riemann_hllc;
  
  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(U);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(U);

  } else {

    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - implode" << std::endl;
    init_implode(U);

  }
  std::cout << "##########################" << "\n";
  std::cout << "Solver is " << m_solver_name << "\n";
  std::cout << "Problem (init condition) is " << m_problem_name << "\n";
  std::cout << "##########################" << "\n";

  // print parameters on screen
  params.print();
  std::cout << "##########################" << "\n";

  // initialize time step
  compute_dt();

  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2,U);

} // SolverHydroMuscl2D::SolverHydroMuscl2D


// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl2D::~SolverHydroMuscl2D()
{

} // SolverHydroMuscl2D::~SolverHydroMuscl2D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMuscl2D::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  // which array is the current one ?
  if (m_iteration % 2 == 0)
    Udata = U;
  else
    Udata = U2;

  // call device functor
  ComputeDtFunctor computeDtFunctor(params, Udata);
  Kokkos::parallel_reduce(ijsize, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // SolverHydroMuscl2D::compute_dt_local

// =======================================================
// =======================================================
void SolverHydroMuscl2D::next_iteration_impl()
{

  if (m_iteration % 10 == 0) {
    std::cout << "time step=" << m_iteration << std::endl;
  }
  
  // output
  if (params.enableOutput) {
    if ( should_save_solution() ) {
      
      std::cout << "Output results at time t=" << m_t
		<< " step " << m_iteration
		<< " dt=" << m_dt << std::endl;
      
      save_solution();
      
    } // end output
  } // end enable output
  
  // compute new dt
  timers[TIMER_DT]->start();
  compute_dt();
  timers[TIMER_DT]->stop();
  
  // perform one step integration
  godunov_unsplit(m_dt);
  
} // SolverHydroMuscl2D::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMuscl2D::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt);
  } else {
    godunov_unsplit_cpu(U2, U , dt);
  }
  
} // SolverHydroMuscl2D::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
void SolverHydroMuscl2D::godunov_unsplit_cpu(DataArray data_in, 
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
    
    // compute fluxes
    {
      ComputeAndStoreFluxesFunctor functor(params, Q,
					   Fluxes_x, Fluxes_y,
					   dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }

    // actual update
    {
      UpdateFunctor functor(params, data_out,
			    Fluxes_x, Fluxes_y);
      Kokkos::parallel_for(ijsize, functor);
    }
    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor computeSlopesFunctor(params, Q, Slopes_x, Slopes_y);
    Kokkos::parallel_for(ijsize, computeSlopesFunctor);

    // now trace along X axis
    {
      ComputeTraceAndFluxes_Functor<XDIR> functor(params, Q,
						  Slopes_x, Slopes_y,
						  Fluxes_x,
						  dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }
    
    // and update along X axis
    {
      UpdateDirFunctor<XDIR> functor(params, data_out, Fluxes_x);
      Kokkos::parallel_for(ijsize, functor);
    }

    // now trace along Y axis
    {
      ComputeTraceAndFluxes_Functor<YDIR> functor(params, Q,
						  Slopes_x, Slopes_y,
						  Fluxes_y,
						  dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }
    
    // and update along Y axis
    {
      UpdateDirFunctor<YDIR> functor(params, data_out, Fluxes_y);
      Kokkos::parallel_for(ijsize, functor);
    }

  } // end params.implementationVersion == 1
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMuscl2D::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMuscl2D::convertToPrimitives(DataArray Udata)
{

  // call device functor
  ConvertToPrimitivesFunctor convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(ijsize, convertToPrimitivesFunctor);
  
} // SolverHydroMuscl2D::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverHydroMuscl2D::make_boundaries(DataArray Udata)
{
  const int ghostWidth=params.ghostWidth;
  int nbIter = ghostWidth*std::max(isize,jsize);
  
  // call device functor
  {
    MakeBoundariesFunctor<FACE_XMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor<FACE_XMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }

  {
    MakeBoundariesFunctor<FACE_YMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor<FACE_YMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  
} // SolverHydroMuscl2D::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void SolverHydroMuscl2D::init_implode(DataArray Udata)
{

  InitImplodeFunctor functor(params, Udata);
  Kokkos::parallel_for(ijsize, functor);
  
} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
void SolverHydroMuscl2D::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);
  
  InitBlastFunctor functor(params, blastParams, Udata);
  Kokkos::parallel_for(ijsize, functor);

} // SolverHydroMuscl2D::init_blast

// =======================================================
// =======================================================
void SolverHydroMuscl2D::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    saveVTK(U, m_times_saved, "U");
  else
    saveVTK(U2, m_times_saved, "U");
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroMuscl2D::save_solution_impl()

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void SolverHydroMuscl2D::saveVTK(DataArray Udata,
				 int iStep,
				 std::string name)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int imin = params.imin;
  const int imax = params.imax;
  const int jmin = params.jmin;
  const int jmax = params.jmax;
  const int ghostWidth = params.ghostWidth;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,iVar;
  std::string outputDir    = configMap.getString("output", "outputDir", "./");
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
  // check scalar data type
  bool useDouble = false;

  if (sizeof(real_t) == sizeof(double)) {
    useDouble = true;
  }
  
  // write iStep in string stepNum
  std::ostringstream stepNum;
  stepNum.width(7);
  stepNum.fill('0');
  stepNum << iStep;
  
  // concatenate file prefix + file number + suffix
  std::string filename     = outputDir + "/" + outputPrefix+"_"+stepNum.str() + ".vti";
  
  // open file 
  std::fstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out);
  
  // write header
  outFile << "<?xml version=\"1.0\"?>\n";
  if (isBigEndian()) {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  } else {
    outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  }

  // write mesh extent
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 0  << " "
	  <<  "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << 1  << " "    
	  << "\">\n";
  
  outFile << "    <PointData>\n";
  outFile << "    </PointData>\n";
  outFile << "    <CellData>\n";

  // write data array (ascii), remove ghost cells
  for ( iVar=0; iVar<nbvar; iVar++) {
    outFile << "    <DataArray type=\"";
    if (useDouble)
      outFile << "Float64";
    else
      outFile << "Float32";
    outFile << "\" Name=\"" << m_variables_names[iVar] << "\" format=\"ascii\" >\n";

    for (int index=0; index<ijsize; ++index) {
      //index2coord(index,i,j,isize,jsize);

      // enforce the use of left layout (Ok for CUDA)
      // but for OpenMP, we will need to transpose
      j = index / isize;
      i = index - j*isize;

      if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	  i>=imin+ghostWidth and i<=imax-ghostWidth) {
    	outFile << Uhost(i, j, iVar) << " ";
      }
    }
    outFile << "\n    </DataArray>\n";
  } // end for iVar

  outFile << "    </CellData>\n";

  // write footer
  outFile << "  </Piece>\n";
  outFile << "  </ImageData>\n";
  outFile << "</VTKFile>\n";
  
  outFile.close();

} // SolverHydroMuscl2D::saveVTK

} // namespace ppkMHD
