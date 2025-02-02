/**
 * \file PapiTest.cpp
 * \brief This is an example use of class PapiInfo.
 *
 * \date March 4, 2014
 * \author Pierre Kestener
 *
 */

#include <iostream>
#include <fstream>
#include "utils/monitoring/PapiInfo.h"

namespace ppkMHD
{

class PapiInfoTest
{
public:
  PapiInfoTest(){};
  ~PapiInfoTest(){};

  OpenMPTimer aTimer;
  PapiInfo    papiFlops_total;

  void
  run()
  {
    double tmp = 1.1;

    papiFlops_total.start();
    {

      for (int i = 1; i < 2000; i++)
      {
        tmp = (tmp + 100) / i;
      }
    }
    papiFlops_total.stop();

    printf("tmp : %g \n", tmp);
    printf("time elapsed : %f\n", papiFlops_total.elapsed());
    printf("total flpops %lld\n", papiFlops_total.getFlop());
    printf("MFLOPS %g\n", 1e-6 * papiFlops_total.getFlop() / papiFlops_total.elapsed());
  };
};

} // namespace ppkMHD


int
main(int argc, char * argv[])
{
  ppkMHD::PapiInfoTest test;

  test.run();

  return 0;
}
