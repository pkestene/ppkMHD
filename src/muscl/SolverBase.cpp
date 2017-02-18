#include "SolverBase.h"

#include "utils.h"

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

} // SolverRunbase::SolverRunbase

// =======================================================
// =======================================================
SolverBase::~SolverBase()
{

} // SolverRunbase::~SolverRunbase

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
  
  m_solver_name = configMap.getString("run", "solver_name", "muscl_2d");

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
  
} // SolverRunbase::finished

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

} // SolverRunbase::next_iteration

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

  // negative value means : no output will ever be done
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
