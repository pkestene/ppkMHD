#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "HydroRun3D.h"
#include "HydroParams.h"
#include "Timer.h"

// the actual computational functors called in HydroRun
#include "HydroRunFunctors3D.h"

// Kokkos
#include "kokkos_shared.h"

static bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}


// =======================================================
// =======================================================
/**
 *
 */
HydroRun3D::HydroRun3D(HydroParams& params, ConfigMap& configMap) :
  params(params),
  configMap(configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  Slopes_x(), Slopes_y(), Slopes_z(),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  ijsize(params.isize*params.jsize),
  ijksize(params.isize*params.jsize*params.ksize)
{

  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", ijksize, nbvar);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",ijksize, nbvar);
  Q     = DataArray("Q", ijksize, nbvar);

  if (params.implementationVersion == 0) {

    Fluxes_x = DataArray("Fluxes_x", ijksize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", ijksize, nbvar);
    Fluxes_z = DataArray("Fluxes_z", ijksize, nbvar);
    
  } else if (params.implementationVersion == 1) {

    Slopes_x = DataArray("Slope_x", ijksize, nbvar);
    Slopes_y = DataArray("Slope_y", ijksize, nbvar);
    Slopes_z = DataArray("Slope_z", ijksize, nbvar);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", ijksize, nbvar);
    Fluxes_y = Fluxes_x;
    Fluxes_z = Fluxes_x;
    
  }
  
  // default riemann solver
  // riemann_solver_fn = &HydroRun3D::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &HydroRun3D::riemann_hllc;
  
  /*
   * initialize hydro array at t=0
   */
  if ( params.problemType == PROBLEM_IMPLODE) {

    init_implode(U);

  } else if (params.problemType == PROBLEM_BLAST) {

    init_blast(U);

  } else {

    std::cout << "Problem : " << params.problemType
	      << " is not recognized / implemented in initHydroRun."
	      << std::endl;
    std::cout <<  "Use default - implode" << std::endl;
    init_implode(U);

  }

  // copy U into U2
  Kokkos::deep_copy(U2,U);

} // HydroRun3D::HydroRun3D


// =======================================================
// =======================================================
/**
 *
 */
HydroRun3D::~HydroRun3D()
{

} // HydroRun3D::~HydroRun3D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \param[in] useU integer, if 0 use data in U else use U2
 *
 * \return dt time step
 */
real_t HydroRun3D::compute_dt(int useU)
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  // which array is the current one ?
  if (useU == 0)
    Udata = U;
  else
    Udata = U2;

  // call device functor
  ComputeDtFunctor3D computeDtFunctor(params, Udata);
  Kokkos::parallel_reduce(ijksize, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // HydroRun3D::compute_dt

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void HydroRun3D::godunov_unsplit(int nStep, real_t dt)
{
  
  if ( nStep % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt, nStep);
  } else {
    godunov_unsplit_cpu(U2, U , dt, nStep);
  }
  
} // HydroRun3D::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
void HydroRun3D::godunov_unsplit_cpu(DataArray data_in, 
				   DataArray data_out, 
				   real_t dt, 
				   int nStep)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

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
    
    // compute fluxes
    {
      ComputeAndStoreFluxesFunctor3D functor(params, Q,
					   Fluxes_x, Fluxes_y, Fluxes_z,
					   dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }

    // actual update
    {
      UpdateFunctor3D functor(params, data_out,
			    Fluxes_x, Fluxes_y, Fluxes_z);
      Kokkos::parallel_for(ijksize, functor);
    }
    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor3D computeSlopesFunctor(params, Q,
					      Slopes_x, Slopes_y, Slopes_z);
    Kokkos::parallel_for(ijksize, computeSlopesFunctor);

    // now trace along X axis
    {
      ComputeTraceAndFluxes_Functor3D<XDIR> functor(params, Q,
						  Slopes_x, Slopes_y, Slopes_z,
						  Fluxes_x,
						  dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }
    
    // and update along X axis
    {
      UpdateDirFunctor3D<XDIR> functor(params, data_out, Fluxes_x);
      Kokkos::parallel_for(ijksize, functor);
    }

    // now trace along Y axis
    {
      ComputeTraceAndFluxes_Functor3D<YDIR> functor(params, Q,
						  Slopes_x, Slopes_y, Slopes_z,
						  Fluxes_y,
						  dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }
    
    // and update along Y axis
    {
      UpdateDirFunctor3D<YDIR> functor(params, data_out, Fluxes_y);
      Kokkos::parallel_for(ijksize, functor);
    }

    // now trace along Z axis
    {
      ComputeTraceAndFluxes_Functor3D<ZDIR> functor(params, Q,
						  Slopes_x, Slopes_y, Slopes_z,
						  Fluxes_z,
						  dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }
    
    // and update along Z axis
    {
      UpdateDirFunctor3D<ZDIR> functor(params, data_out, Fluxes_z);
      Kokkos::parallel_for(ijksize, functor);
    }

  } // end params.implementationVersion == 1
  
  godunov_timer.stop();
  
} // HydroRun3D::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void HydroRun3D::convertToPrimitives(DataArray Udata)
{

  // call device functor
  ConvertToPrimitivesFunctor3D convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(ijksize, convertToPrimitivesFunctor);
  
} // HydroRun3D::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void HydroRun3D::make_boundaries(DataArray Udata)
{
  const int ghostWidth=params.ghostWidth;

  int max_size = std::max(isize,jsize);
  max_size = std::max(max_size,ksize);
  int nbIter = ghostWidth * max_size * max_size;
  
  // call device functor
  {
    MakeBoundariesFunctor3D<FACE_XMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }  
  {
    MakeBoundariesFunctor3D<FACE_XMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor3D<FACE_YMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor3D<FACE_YMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor3D<FACE_ZMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor3D<FACE_ZMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  
} // HydroRun3D::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void HydroRun3D::init_implode(DataArray Udata)
{

  InitImplodeFunctor3D functor(params, Udata);
  Kokkos::parallel_for(ijksize, functor);

} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
void HydroRun3D::init_blast(DataArray Udata)
{

  InitBlastFunctor3D functor(params, Udata);
  Kokkos::parallel_for(ijksize, functor);

} // HydroRun3D::init_blast

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////
// output routine (VTK file format, ASCII, VtkImageData)
// Take care that VTK uses row major (i+j*nx+k*nx*ny)
// To make sure OpenMP and CUDA version give the same
// results, we transpose the OpenMP data.
// ///////////////////////////////////////////////////////
void HydroRun3D::saveVTK(DataArray Udata,
			 int iStep,
			 std::string name)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nz = params.nz;
  const int imin = params.imin;
  const int imax = params.imax;
  const int jmin = params.jmin;
  const int jmax = params.jmax;
  const int kmin = params.kmin;
  const int kmax = params.kmax;
  const int ghostWidth = params.ghostWidth;
  
  // copy device data to host
  Kokkos::deep_copy(Uhost, Udata);
  
  // local variables
  int i,j,k,iVar;
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
#ifdef CUDA
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nz  << " "
	  <<  "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nx << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nz  << " "    
	  << "\">\n";
#else
  outFile << "  <ImageData WholeExtent=\""
	  << 0 << " " << nz << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nx  << " "
	  <<  "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
  outFile << "  <Piece Extent=\""
	  << 0 << " " << nz << " "
	  << 0 << " " << ny << " "
	  << 0 << " " << nx  << " "    
	  << "\">\n";
#endif
  
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
    outFile << "\" Name=\"" << varNames[iVar] << "\" format=\"ascii\" >\n";
    
    for (int index=0; index<ijksize; ++index) {
      index2coord(index,i,j,k,isize,jsize,ksize);
      if (k>=kmin+ghostWidth and k<=kmax-ghostWidth and
	  j>=jmin+ghostWidth and j<=jmax-ghostWidth and
	  i>=imin+ghostWidth and i<=imax-ghostWidth) {
	outFile << Uhost(index , iVar) << " ";
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
  
} // HydroRun3D::saveVTK

