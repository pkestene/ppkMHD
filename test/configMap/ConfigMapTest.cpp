// Example that shows simple usage of the INIReader class

#include <iostream>
#include <fstream>
#include "ConfigMap.h"

int main(int argc, char* argv[])
{
  // make test.ini file
  std::fstream iniFile;
  iniFile.open ("./test.ini", std::ios_base::out);
  iniFile << "; Test config file for ini_test.c" << std::endl;
  
  iniFile << "[Protocol]             ; Protocol configuration" << std::endl;
  iniFile << "Version=6              ; IPv6" << std::endl;
  
  iniFile << "[User]" << std::endl;
  iniFile << "Name = Bob Smith       ; Spaces around '=' are stripped" << std::endl;
  iniFile << "Email = bob@smith.com  ; And comments (like this) ignored" << std::endl;
  iniFile.close();

  // create a ConfigMap instance
  ConfigMap configMap("./test.ini");
  
  if (configMap.ParseError() < 0) {
    std::cout << "Can't load 'test.ini'\n";
    return 1;
  }
  std::cout << "Config loaded from 'test.ini': version="
	    << configMap.getInteger("protocol", "version", -1) << ", name="
	    << configMap.getString("user", "name", "UNKNOWN") << ", email="
	    << configMap.getString("user", "email", "UNKNOWN") << "\n";
  
  ConfigMap configMap2 = configMap;
  std::cout << std::endl;
  std::cout << "Config copied from configMap: version="
	    << configMap.getInteger("protocol", "version", -1) << ", name="
	    << configMap.getString("user", "name", "UNKNOWN") << ", email="
	    << configMap.getString("user", "email", "UNKNOWN") << "\n";
  

  return 0;
}
