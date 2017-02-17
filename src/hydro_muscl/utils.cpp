#include "utils.h"

#include <ctime>   // for std::time_t, std::tm, std::localtime
#include <sstream> // string stream
#include <string>  // string
#include <iomanip> // for std::put_time
#include <iostream>

// =======================================================
// =======================================================
void print_current_date(std::ostream& stream)
{
  
  /* get current time */
  std::time_t     now = std::time(nullptr); 
  
  /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
  std::tm tm = *std::localtime(&now);
  
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %Z");

  const std::string tmp = ss.str();
  //const char *cstr = tmp.c_str();

  stream << "-- " << tmp << "\n";
  
} // print_current_date
