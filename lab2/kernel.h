/*
 * MACA Kernels
 * Dezheng Yan, 2024
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <mc_runtime.h>
#include <vector>

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */
typedef struct {
  float x, y, z, vx, vy, vz;
} Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remains a host function.
 */
__global__ void body_force_kernel(Body *p, float dt, int n);

__global__ void integrate_position_kernel(Body *p, float dt, int n);

void call_body_force_kernel(Body *p, float dt, int n);

void call_integrate_position_kernel(Body *p, float dt, int n);

#endif // KERNEL_H