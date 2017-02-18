#ifndef SOLVER_BASE_H_
#define SOLVER_BASE_H_

#include "HydroParams.h"
#include "config/ConfigMap.h"

#include <map>
#include <memory>

// for timer
#ifdef CUDA
#include "CudaTimer.h"
#else
#include "OpenMPTimer.h"
#endif

enum TimerIds {
  TIMER_TOTAL = 0,
  TIMER_IO = 1,
  TIMER_DT = 2,
  TIMER_BOUNDARIES = 3,
  TIMER_NUM_SCHEME = 4
}; // enum TimerIds

/**
 * Abstract base class for all our actual solvers.
 */
class SolverBase {
  
public:
  
  SolverBase(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverBase();

  // hydroParams
  HydroParams& params;
  ConfigMap& configMap;

  /* some common member data */

  //! is this a restart run ?
  int m_restart_run_enabled;

  //! filename containing data from a previous run.
  std::string m_restart_run_filename;
  
  // iteration info
  double               m_t;         //!< the time at the current iteration
  double               m_dt;        //!< the time step at the current iteration
  int                  m_iteration; //!< the current iteration (integer)
  double               m_tEnd;      //!< maximun time
  double               m_cfl;       //!< Courant number

  long long int        m_nCells;    //!< number of cells

  //! init condition name (or problem)
  std::string          m_problem_name;
  
  //! solver name (use in output file).
  std::string          m_solver_name;

  /*
   *
   * Computation interface that may be overriden in a derived 
   * concrete implementation.
   *
   */

  //! Read and parse the configuration file (ini format).
  virtual void read_config();

  //! Compute CFL condition (allowed time step), over all MPI process.
  virtual void compute_dt();

  //! Compute CFL condition local to current MPI process
  virtual double compute_dt_local();

  //! Check if current time is larger than end time.
  virtual int finished();

  //! This is where action takes place. Wrapper arround next_iteration_impl.
  virtual void next_iteration();

  //! This is the next iteration computation (application specific).
  virtual void next_iteration_impl();
  
  //! Decides if the current time step is eligible for dump data to file
  virtual int  should_save_solution();

  //! main routine to dump solution to file
  virtual void save_solution();

  //! main routine to dump solution to file
  virtual void save_solution_impl();

  //! read restart data
  virtual void read_restart_file();
  
  /* IO related */

  //! counter incremented each time an output is written
  int m_times_saved;

  //! Number of variables to saved
  //int m_write_variables_number;

  //! names of variables to save
  std::map<int, std::string> m_variables_names;

  //! timers
#ifdef CUDA
  using Timer = CudaTimer;
#else
  using Timer = OpenMPTimer;
#endif
  using TimerMap = std::map<int, std::shared_ptr<Timer> >;
  TimerMap timers;

}; // class SolverBase

#endif // SOLVER_BASE_H_
