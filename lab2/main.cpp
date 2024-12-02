#include <mc_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "check.h"
#include "helper/input.h"
#include "helper/output.h"
#include "kernel.h"
#include "timer.h"

// #define DEBUG_WRITE_TO_FILE

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    float tmp = 2.0f * (std::rand() / (float)RAND_MAX) - 1.0f;
    data[i] = std::stof(std::to_string(
        tmp));  // make sure data[i] equals to number saved as test case
  }
}

int main(const int argc, const char **argv) {
  const bool oj_mode = argc > 1;  // read input data from args, write output
                                  // data to file
  /*
   * Do not change the value for `nBodies` here. If you would like to modify it,
   * pass values into the command line.
   */

  int nBodies = 2 << 11;
  int salt = 0;

  const float dt = 0.01f;  // time step
  const int nIters = 10;   // Do not modify it (simulation iterations)

  const int INPUT_ARGS_NUM = 2;

  int bytes = nBodies * sizeof(Body);
  float *buf;

  // buf = (float *)malloc(bytes);
  mcMallocManaged(&buf, bytes);
  Body *p = (Body *)buf;

  /*
   * As a constraint of this exercise, `randomizeBodies` must remain a host
   * function.
   */

  const int size = nBodies * 6;
  if (oj_mode) {
    ReadInputData(buf, size);
  } else {
    randomizeBodies(buf, size);  // Init pos / vel data
#ifdef DEBUG_WRITE_TO_FILE
    WriteFile("1-input.txt", buf, size);
#endif
  }

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  /*******************************************************************/
  // Do not modify the code in this section.
  double totalTime = 0.0;
  double minTime = std::numeric_limits<double>::max();
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();
    /*******************************************************************/

    /*
     * You will likely wish to refactor the work being done in `bodyForce`,
     * as well as the work to integrate the positions.
     */

    // compute interbody forces
    call_body_force_kernel(p, dt, nBodies);  

    /*
     * This position integration cannot occur until this round of `bodyForce`
     * has completed. Also, the next round of `bodyForce` cannot begin until the
     * integration is complete.
     */
    call_integrate_position_kernel(p, dt, nBodies);
    if (iter == nIters - 1) {
      mcDeviceSynchronize();
    }

    /*******************************************************************/
    // Do not modify the code in this section.
    const double tElapsed = GetTimer() / 1000.0;
    printf("iter=%d: tElapsed is %0.6f second\n", iter, tElapsed);
    totalTime += tElapsed;
    if (minTime > tElapsed) {
      minTime = tElapsed;
    }
    /*******************************************************************/
  }

  if (oj_mode) {
    OutputResult(argv[1], buf, size);
  } else {
    checkAccuracy(buf, nBodies);
#ifdef DEBUG_WRITE_TO_FILE
    WriteFile("1-output.txt", buf, size);
#endif
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  printf("nIters=%d: totalTime is %0.6f, avgTime is %0.6f, minTime is %0.6f\n",
         nIters, totalTime, avgTime, minTime);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies,
         billionsOfOpsPerSecond);
  salt += 1;
  /*******************************************************************/

  /*
   * Feel free to modify code below.
   */

  mcFree(buf);
}
