#include <mc_runtime.h>

#include <cstdio>
#include <vector>

#include "check.h"
#include "helper/input.h"
#include "helper/output.h"
#include "helper_maca.h"
#include "kernel.h"
#include "maca_allocator.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "timer.h"

template <class A>
std::tuple<int, int, int> read_image(A &a, const char *path) {
  int nx = 0, ny = 0, comp = 0;
  unsigned char *p = stbi_load(path, &nx, &ny, &comp, 0);
  if (!p) {
    perror(path);
    exit(-1);
  }
  a.resize(nx * ny * comp);
  for (int c = 0; c < comp; c++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        a[c * nx * ny + y * nx + x] =
            (1.f / 255.f) * p[(y * nx + x) * comp + c];
      }
    }
  }
  stbi_image_free(p);
  return {nx, ny, comp};
}

template <class A>
std::tuple<int, int, int> read_image_from_memory(
    A &a, const std::vector<unsigned char> &data) {
  int nx = 0, ny = 0, comp = 0;
  unsigned char *p =
      stbi_load_from_memory(data.data(), data.size(), &nx, &ny, &comp, 0);
  if (!p) {
    perror("load from memory failed");
    exit(-1);
  }
  a.resize(nx * ny * comp);
  for (int c = 0; c < comp; c++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        a[c * nx * ny + y * nx + x] =
            (1.f / 255.f) * p[(y * nx + x) * comp + c];
      }
    }
  }
  stbi_image_free(p);
  return {nx, ny, comp};
}

template <class A>
void write_image(A const &a, int nx, int ny, int comp, const char *path) {
  auto p = (unsigned char *)malloc(nx * ny * comp);
  for (int c = 0; c < comp; c++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        p[(y * nx + x) * comp + c] =
            std::max(0.f, std::min(255.f, a[c * nx * ny + y * nx + x] * 255.f));
      }
    }
  }
  int ret = 0;
  auto pt = strrchr(path, '.');
  if (pt && !strcmp(pt, ".png")) {
    ret = stbi_write_png(path, nx, ny, comp, p, 0);
  } else if (pt && !strcmp(pt, ".jpg")) {
    ret = stbi_write_jpg(path, nx, ny, comp, p, 0);
  } else {
    ret = stbi_write_bmp(path, nx, ny, comp, p);
    /* ret = stbi_write_jpg(path, nx, ny, comp, p, 0); */
  }
  free(p);
  if (!ret) {
    perror(path);
    exit(-1);
  }
}

int main(const int argc, const char **argv) {
  const bool oj_mode = argc > 1;  // read input data from args, write output
                                  // data to file

  const int nIters = 10;  // repeat iterations
  std::vector<float, MacaAllocator<float>> in;
  std::vector<float, MacaAllocator<float>> out;

  int nx, ny, _;
  if (oj_mode) {
    std::istreambuf_iterator<char> it(std::cin);
    std::istreambuf_iterator<char> end;
    std::vector<unsigned char> buffer(it, end);
    std::tie(nx, ny, _) = read_image_from_memory(in, buffer);
  } else {
    std::tie(nx, ny, _) = read_image(in, "../resources/1-input.bmp");
  }
  out.resize(in.size());

  double totalTime = 0.0;
  double minTime = std::numeric_limits<double>::max();
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    call_image_filtering_kernel(out.data(), in.data(), nx, ny);
    mcDeviceSynchronize();

    const double tElapsed = GetTimer() / 1000.0;
    printf("iter=%d: tElapsed is %0.6f second\n", iter, tElapsed);
    totalTime += tElapsed;
    if (minTime > tElapsed) {
      minTime = tElapsed;
    }
  }
  double avgTime = totalTime / (double)(nIters);
  printf("nIters=%d: totalTime is %0.6f, avgTime is %0.6f, minTime is %0.6f\n",
         nIters, totalTime, avgTime, minTime);

  if (oj_mode) {
    write_image(out, nx, ny, 1, argv[1]);
  } else {
    const char *my_out_file_name = "../resources/my-output.bmp";
    write_image(out, nx, ny, 1, my_out_file_name);

    // then, my out should be same as expected_out
    std::vector<float> my_out;
    read_image(my_out, my_out_file_name);

    std::vector<float> expected_out;
    read_image(expected_out, "../resources/1-output.bmp");

    assert(my_out.size() == expected_out.size());
    for (int i = 0; i < my_out.size(); ++i) {
      if (!MetaXOJ::FloatNearlyEqual(my_out[i], expected_out[i])) {
        printf("bad result, my_out[%d](%f) != expected_out[%d](%f)\n", i,
               my_out[i], i, expected_out[i]);
        break;
      }
    }
  }

  return 0;
}
