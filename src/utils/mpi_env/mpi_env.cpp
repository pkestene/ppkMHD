#include "mpi_env.h"

#include <cassert>
#include <string>
#include <exception>
#include <stdexcept>
#include <ostream>



namespace ppkMHD { namespace mpi {

/**
 * Call the MPI routine MPIFunc with arguments Args (surrounded by
 * parentheses). If the result is not MPI_SUCCESS, just print to
 * stderr.
 * 
 */
#define MY_MPI_CHECK_RESULT( MPIFunc, Args )                         \
 {                                                                      \
   int _check_result = MPIFunc Args;                                    \
   if (_check_result != MPI_SUCCESS)                                    \
     std::cerr << "[ppkMHd Error in] " << #MPIFunc << " with result : " << _check_result << "\n"; \
 }


namespace threading {

std::istream& operator>>(std::istream& in, level& l)
{
  std::string tk;
  in >> tk;
  if (!in.bad()) {
    if (tk == "single") {
      l = single;
    } else if (tk == "funneled") {
      l = funneled;
    } else if (tk == "serialized") {
      l = serialized;
    } else if (tk == "multiple") {
      l = multiple;
    } else {
      in.setstate(std::ios::badbit);
    }
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, level l)
{
  switch(l) {
  case single:
    out << "single";
    break;
  case funneled:
    out << "funneled";
    break;
  case serialized:
    out << "serialized";
    break;
  case multiple:
    out << "multiple";
    break;
  default:
    out << "<level error>[" << int(l) << ']';
    out.setstate(std::ios::badbit);
    break;
  }
  return out;
}

} // namespace threading

environment::environment(int& argc, char** &argv, bool abort_on_exception)
  : i_initialized(false),
    abort_on_exception(abort_on_exception)
{
  if (!initialized()) {
    MY_MPI_CHECK_RESULT(MPI_Init, (&argc, &argv));
    i_initialized = true;
  }

#if (2 <= MPI_VERSION)
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#else
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#endif
}

environment::environment(int& argc, char** &argv, threading::level mt_level,
                         bool abort_on_exception)
  : i_initialized(false),
    abort_on_exception(abort_on_exception)
{
  // It is not clear that we can pass null in MPI_Init_thread.
  int dummy_thread_level = 0;
  if (!initialized()) {
    MY_MPI_CHECK_RESULT(MPI_Init_thread, 
                           (&argc, &argv, int(mt_level), &dummy_thread_level));
    i_initialized = true;
  }

#if (2 <= MPI_VERSION)
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#else
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#endif
}

environment::~environment()
{
  if (i_initialized) {
    if (std::uncaught_exception() && abort_on_exception) {
      abort(-1);
    } else if (!finalized()) {
      MY_MPI_CHECK_RESULT(MPI_Finalize, ());
    }
  }
}

void environment::abort(int errcode)
{
  MY_MPI_CHECK_RESULT(MPI_Abort, (MPI_COMM_WORLD, errcode));
}

bool environment::initialized()
{
  int flag;
  MY_MPI_CHECK_RESULT(MPI_Initialized, (&flag));
  return flag != 0;
}

bool environment::finalized()
{
  int flag;
  MY_MPI_CHECK_RESULT(MPI_Finalized, (&flag));
  return flag != 0;
}

int environment::max_tag()
{
  int* max_tag_value;
  int found = 0;

#if (2 <= MPI_VERSION)
  MY_MPI_CHECK_RESULT(MPI_Comm_get_attr,
                         (MPI_COMM_WORLD, MPI_TAG_UB, &max_tag_value, &found));
#else
  MY_MPI_CHECK_RESULT(MPI_Attr_get,
                         (MPI_COMM_WORLD, MPI_TAG_UB, &max_tag_value, &found));
#endif
  assert(found != 0);
  return *max_tag_value - num_reserved_tags;
}

int environment::collectives_tag()
{
  return max_tag() + 1;
}

std::string environment::processor_name()
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int len;

  MY_MPI_CHECK_RESULT(MPI_Get_processor_name, (name, &len));
  return std::string(name, len);
}

threading::level environment::thread_level()
{
  int level;

  MY_MPI_CHECK_RESULT(MPI_Query_thread, (&level));
  return static_cast<threading::level>(level);
}

bool environment::is_main_thread()
{
  int isit;

  MY_MPI_CHECK_RESULT(MPI_Is_thread_main, (&isit));
  return static_cast<bool>(isit);
}

std::pair<int, int> environment::version()
{
  int major, minor;
  MY_MPI_CHECK_RESULT(MPI_Get_version, (&major, &minor));
  return std::make_pair(major, minor);
}

} } // end namespace ppkMHD::mpi
