#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "SolverMHDMuscl2D.h"
#include "HydroParams.h"

// the actual computational functors called in SolverMHDMuscl2D
#include "MHDRunFunctors2D.h"

// Kokkos
#include "kokkos_shared.h"

static bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}

// =======================================================
// ==== CLASS SolverMHDMuscl2D IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverMHDMuscl2D::SolverMHDMuscl2D(HydroParams& params, ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Qm_x(), Qm_y(),
  Qp_x(), Qp_y(),
  QEdge_RT(), QEdge_RB(),
  QEdge_LT(), QEdge_LB(),
  Fluxes_x(), Fluxes_y(),
  Emf(),
  isize(params.isize),
  jsize(params.jsize),
  ijsize(params.isize*params.jsize)
{

  m_nCells = ijsize;

  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", ijsize);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",ijsize);
  Q     = DataArray("Q", ijsize);

  if (params.implementationVersion == 0) {
  
    Qm_x = DataArray("Qm_x", ijsize);
    Qm_y = DataArray("Qm_y", ijsize);
    Qp_x = DataArray("Qp_x", ijsize);
    Qp_y = DataArray("Qp_y", ijsize);

    QEdge_RT = DataArray("QEdge_RT", ijsize);
    QEdge_RB = DataArray("QEdge_RB", ijsize);
    QEdge_LT = DataArray("QEdge_LT", ijsize);
    QEdge_LB = DataArray("QEdge_LB", ijsize);
    
    Fluxes_x = DataArray("Fluxes_x", ijsize);
    Fluxes_y = DataArray("Fluxes_y", ijsize);

    Emf = DataArrayScalar("Emf", ijsize);
    
  }
  
  // default riemann solver
  // riemann_solver_fn = &SolverMHDMuscl2D::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &SolverMHDMuscl2D::riemann_hllc;
  
  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name == "orszag_tang") {

    init_orszag_tang(U);

  } else {

    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - orszag_tang" << std::endl;
    init_orszag_tang(U);

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

  // upload Uhost to device
  Kokkos::deep_copy(U2,U);

} // SolverMHDMuscl2D::SolverMHDMuscl2D


// =======================================================
// =======================================================
/**
 *
 */
SolverMHDMuscl2D::~SolverMHDMuscl2D()
{

} // SolverMHDMuscl2D::~SolverMHDMuscl2D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \param[in] useU integer, if 0 use data in U else use U2
 *
 * \return dt time step
 */
real_t SolverMHDMuscl2D::compute_dt_local()
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

} // SolverMHDMuscl2D::compute_dt_local

// =======================================================
// =======================================================
void SolverMHDMuscl2D::next_iteration_impl()
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
  
} // SolverMHDMuscl2D::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverMHDMuscl2D::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt);
  } else {
    godunov_unsplit_cpu(U2, U , dt);
  }
  
} // SolverMHDMuscl2D::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
void SolverMHDMuscl2D::godunov_unsplit_cpu(DataArray data_in, 
					   DataArray data_out, 
					   real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

  // fill ghost cell in data_in
  boundaries_timer.start();
  make_boundaries(data_in);
  boundaries_timer.stop();
    
  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  godunov_timer.start();

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  if (params.implementationVersion == 0) {
    
    // trace computation: fill arrays qm_x, qm_y, qp_x, qp_y
    computeTrace(data_in, dt);

    // Compute flux via Riemann solver and update (time integration)
    computeFluxesAndStore(dt);

    // Compute Emf
    computeEmfAndStore(dt);
    
    // actual update with fluxes
    {
      UpdateFunctor functor(params, data_out,
			    Fluxes_x, Fluxes_y, dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }

    // actual update with emf
    {
      UpdateEmfFunctor functor(params, data_out,
			       Emf, dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }
    
  }
  godunov_timer.stop();
  
} // SolverMHDMuscl2D::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::convertToPrimitives(DataArray Udata)
{

  // call device functor
  ConvertToPrimitivesFunctor convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(ijsize, convertToPrimitivesFunctor);
  
} // SolverMHDMuscl2D::convertToPrimitives

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Compute trace (only used in implementation version 2), i.e.
// fill global array qm_x, qmy, qp_x, qp_y
// ///////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::computeTrace(DataArray Udata, real_t dt)
{

  // local variables
  real_t dtdx;
  real_t dtdy;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

  // call device functor
  ComputeTraceFunctor computeTraceFunctor(params, Udata, Q,
					  Qm_x, Qm_y,
					  Qp_x, Qp_y,
					  QEdge_RT, QEdge_RB,
					  QEdge_LT, QEdge_LB,
					  dtdx, dtdy);
  Kokkos::parallel_for(ijsize, computeTraceFunctor);

} // SolverMHDMuscl2D::computeTrace

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute flux via Riemann solver and store
// //////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::computeFluxesAndStore(real_t dt)
{
   
  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeFluxesAndStoreFunctor
    computeFluxesAndStoreFunctor(params,
				 Qm_x, Qm_y,
				 Qp_x, Qp_y,
				 Fluxes_x, Fluxes_y,
				 dtdx, dtdy);
  Kokkos::parallel_for(ijsize, computeFluxesAndStoreFunctor);
  
} // computeFluxesAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute EMF via 2D Riemann solver and store
// //////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::computeEmfAndStore(real_t dt)
{
   
  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeEmfAndStoreFunctor
    computeEmfAndStoreFunctor(params,
			      QEdge_RT, QEdge_RB,
			      QEdge_LT, QEdge_LB,
			      Emf,
			      dtdx, dtdy);
  Kokkos::parallel_for(ijsize, computeEmfAndStoreFunctor);
  
} // computeEmfAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverMHDMuscl2D::make_boundaries(DataArray Udata)
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
  
  
} // SolverMHDMuscl2D::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
// void SolverMHDMuscl2D::init_implode(DataArray Udata)
// {

//   InitImplodeFunctor functor(params, Udata);
//   Kokkos::parallel_for(ijsize, functor);
  
// } // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
// void SolverMHDMuscl2D::init_blast(DataArray Udata)
// {

//   InitBlastFunctor functor(params, Udata);
//   Kokkos::parallel_for(ijsize, functor);
  
// } // SolverMHDMuscl2D::init_blast

// =======================================================
// =======================================================
/**
 * Orszag-Tang vortex test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
 */
void SolverMHDMuscl2D::init_orszag_tang(DataArray Udata)
{
  
  // init all vars but energy
  {
    InitOrszagTangFunctor<INIT_ALL_VAR_BUT_ENERGY> functor(params, Udata);
    Kokkos::parallel_for(ijsize, functor);
  }

  // init energy
  {
    InitOrszagTangFunctor<INIT_ENERGY> functor(params, Udata);
    Kokkos::parallel_for(ijsize, functor);
  }  
  
} // init_orszag_tang

// =======================================================
// =======================================================
void SolverMuscl2D::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    saveVTK(U, m_times_saved, "U");
  else
    saveVTK(U2, m_times_saved, "U");
  
  timers[TIMER_IO]->stop();
    
} // SolverMuscl2D::save_solution_impl()

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// ///////////////////////////////////////////////////////
void SolverMHDMuscl2D::saveVTK(DataArray Udata,
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
  for ( iVar=0; iVar<NBVAR; iVar++) {
    outFile << "    <DataArray type=\"";
    if (useDouble)
      outFile << "Float64";
    else
      outFile << "Float32";
    outFile << "\" Name=\"" << varNames[iVar] << "\" format=\"ascii\" >\n";

    for (int index=0; index<ijsize; ++index) {
      //index2coord(index,i,j,isize,jsize);

      // enforce the use of left layout (Ok for CUDA)
      // but for OpenMP, we will need to transpose
      j = index / isize;
      i = index - j*isize;

      if (j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	  i>=imin+ghostWidth and i<=imax-ghostWidth) {
#ifdef CUDA
    	outFile << Uhost(index , iVar) << " ";
#else
	int index2 = j+jsize*i;
    	outFile << Uhost(index2 , iVar) << " ";
#endif
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

} // SolverMHDMuscl2D::saveVTK

