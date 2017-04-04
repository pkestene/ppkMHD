#include "SolverBase.h"

#include "utils.h"

#include <io/IO_Writer.h>

namespace ppkMHD {

// =======================================================
// ==== CLASS SolverBase IMPL ============================
// =======================================================

// =======================================================
// =======================================================
SolverBase::SolverBase (HydroParams& params, ConfigMap& configMap) :
  params(params),
  configMap(configMap)
{

  /*
   * init some variables by reading parameter file.
   */
  read_config();

  /*
   * other variables initialization.
   */
  m_times_saved = 0;
  m_nCells = 0;

  // create the timers
  timers[TIMER_TOTAL]      = std::make_shared<Timer>();
  timers[TIMER_IO]         = std::make_shared<Timer>();
  timers[TIMER_DT]         = std::make_shared<Timer>();
  timers[TIMER_BOUNDARIES] = std::make_shared<Timer>();
  timers[TIMER_NUM_SCHEME] = std::make_shared<Timer>();

  // init variables names
  m_variables_names[ID] = "rho";
  m_variables_names[IP] = "energy";
  m_variables_names[IU] = "mx"; // momentum component X
  m_variables_names[IV] = "my"; // momentum component Y
  m_variables_names[IW] = "mz"; // momentum component Z
  m_variables_names[IA] = "bx"; // mag field X
  m_variables_names[IB] = "by"; // mag field Y
  m_variables_names[IC] = "bz"; // mag field Z

  m_io_writer = new io::IO_Writer(params, configMap, m_variables_names);
  
} // SolverBase::SolverBase

// =======================================================
// =======================================================
SolverBase::~SolverBase()
{

  delete m_io_writer;
  
} // SolverBase::~SolverBase

// =======================================================
// =======================================================
void
SolverBase::read_config()
{

  m_t     = configMap.getFloat("run", "tCurrent", 0.0);
  m_tEnd  = configMap.getFloat("run", "tEnd", 0.0);
  m_dt    = m_tEnd;
  m_cfl   = configMap.getFloat("hydro", "cfl", 1.0);
  m_iteration = 0;

  m_problem_name = configMap.getString("hydro", "problem", "unknown");

  m_solver_name = configMap.getString("run", "solver_name", "unknown");

  /* restart run : default is no */
  m_restart_run_enabled = configMap.getInteger("run", "restart_enabled", 0);
  m_restart_run_filename = configMap.getString ("run", "restart_filename", "");

} // SolverBase::read_config

// =======================================================
// =======================================================
void
SolverBase::compute_dt()
{

#ifdef HAVE_MPI

  // get local time step
  double dt_local = compute_time_step_local();

  // perform MPI_Reduceall to get global time step
  
#else

  m_dt = compute_dt_local();
  
#endif
  
} // SolverBase::compute_dt

// =======================================================
// =======================================================
double
SolverBase::compute_dt_local()
{

  // the actual numerical scheme must provide it a genuine implementation

  return m_tEnd;
  
} // SolverBase::compute_dt_local

// =======================================================
// =======================================================
int
SolverBase::finished()
{

  return m_t >= (m_tEnd - 1e-14) || m_iteration >= params.nStepmax;
  
} // SolverBase::finished

// =======================================================
// =======================================================
void
SolverBase::next_iteration()
{

  // setup a timer here (?)
  
  // genuine implementation called here
  next_iteration_impl();

  // perform some stats here (?)
  
  // incremenent
  ++m_iteration;
  m_t += m_dt;

} // SolverBase::next_iteration

// =======================================================
// =======================================================
void
SolverBase::next_iteration_impl()
{

  // This is application dependent
  
} // SolverBase::next_iteration_impl

// =======================================================
// =======================================================
void
SolverBase::save_solution()
{

  // save solution to output file
  save_solution_impl();
  
  // increment output file number
  ++m_times_saved;
  
} // SolverBase::save_solution

// =======================================================
// =======================================================
void
SolverBase::save_solution_impl()
{
} // SolverBase::save_solution_impl

// =======================================================
// =======================================================
void
SolverBase::read_restart_file()
{

  // TODO
  
} // SolverBase::read_restart_file

// =======================================================
// =======================================================
int
SolverBase::should_save_solution()
{
  
  double interval = m_tEnd / params.nOutput;

  // params.nOutput == 0 means no output at all
  // params.nOutput < 0  means always output 
  if (params.nOutput < 0) {
    return 1;
  }

  if ((m_t - (m_times_saved - 1) * interval) > interval) {
    return 1;
  }

  /* always write the last time step */
  if (ISFUZZYNULL (m_t - m_tEnd)) {
    return 1;
  }

  return 0;
  
} // SolverBase::should_save_solution

// =======================================================
// =======================================================
void
SolverBase::save_data(DataArray2d             U,
		      DataArray2d::HostMirror Uh,
		      int iStep)
{
  m_io_writer->save_data(U, Uh, iStep, "");
}

// =======================================================
// =======================================================
void
SolverBase::save_data(DataArray3d             U,
		      DataArray3d::HostMirror Uh,
		      int iStep)
{
  m_io_writer->save_data(U, Uh, iStep, "");
}

// =======================================================
// =======================================================
void
SolverBase::save_data_debug(DataArray2d             U,
			    DataArray2d::HostMirror Uh,
			    int iStep,
			    std::string debug_name)
{
  m_io_writer->save_data(U, Uh, iStep, debug_name);
}

// =======================================================
// =======================================================
void
SolverBase::save_data_debug(DataArray3d             U,
			    DataArray3d::HostMirror Uh,
			    int iStep,
			    std::string debug_name)
{
  m_io_writer->save_data(U, Uh, iStep, debug_name);
}

} // namespace ppkMHD
