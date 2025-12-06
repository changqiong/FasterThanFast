/**
   This file is part of FasterThanFast. (https://github.com/changqiong/FasterThanFast.git).

   Copyright (c) 2025 Qiong Chang.

   FasterThanFast is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   any later version.

   FasterThanFast is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with FasterThanFast.  If not, see <http://www.gnu.org/licenses/>.
**/

#include "fast_kernel.h"
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#define METHOD 4
#define N_OCTAVES 4


namespace orb
{
  // Kernel configuration
#define X1 64
#define X2 32
  // ORB parameters
#define MAX_OCTAVE 5
#define FAST_PATTERN 16
#define HARRIS_SIZE 7
#define MAX_PATCH 31
#define K (FAST_PATTERN / 2)
#define N (FAST_PATTERN + K + 1)
#define HARRIS_K 0.04f
#define MAX_DIST 64
#define GR 3
#define R2 6
#define R4 12
#define DX (X2 - R2)
#define FAST_WIDTH (X_FAST+8)
#define X_FAST 32
#define FAST_HEIGHT 9

  constexpr int kCirclePoints       = 16;
  constexpr int kContiguousRequired = 9;
  constexpr int kFastThresholdLUTSize = 512;

  __constant__ int d_max_num_points;
  __constant__ float d_scale_sq_sq;
  __device__ unsigned int d_point_counter;
  __constant__ int dpixel[25 * MAX_OCTAVE];
  __constant__ unsigned char dthresh_table[512];
  __constant__ int d_umax[MAX_PATCH / 2 + 2];
  __constant__ int2 d_pattern[512];
  __constant__ float d_gauss[GR + 1];
  __constant__ int ofs[HARRIS_SIZE * HARRIS_SIZE];
  __constant__ int angle_param[MAX_OCTAVE * 2];

  void setMaxKeypointCount(int max_keypoints)
  {
    CHECK(cudaMemcpyToSymbol(d_max_num_points,
                             &max_keypoints,
                             sizeof(int),
                             0,
                             cudaMemcpyHostToDevice));
  }

  void getPointCounterDeviceAddress(void** device_address)
  {
    CHECK(cudaGetSymbolAddress(device_address, d_point_counter));
  }

  void initializeFastThresholdLUT(int fast_threshold)
  {
    unsigned char threshold_lut[kFastThresholdLUTSize];

    for (int intensity_diff = -255, idx = 0; intensity_diff <= 255; ++intensity_diff, ++idx) {
      threshold_lut[idx] =
	static_cast<unsigned char>(intensity_diff < -fast_threshold ? 1
				   : intensity_diff >  fast_threshold ? 2
				   : 0);
    }

    CHECK(cudaMemcpyToSymbol(dthresh_table,
                             threshold_lut,
                             kFastThresholdLUTSize * sizeof(unsigned char),
                             0,
                             cudaMemcpyHostToDevice));
  }

  void initializeUmaxTable(int patch_size)
  {
    const int half_patch = patch_size / 2;
    std::vector<int> umax(half_patch + 2);
    umax[half_patch + 1] = 0;

    const float v        = half_patch * std::sqrt(2.0f) * 0.5f;
    const int   vmax     = static_cast<int>(std::floor(v + 1.0f));
    const int   vmin     = static_cast<int>(std::ceil(v));

    for (int y = 0; y <= vmax; ++y) {
      umax[y] = static_cast<int>(
				 std::round(std::sqrt(static_cast<float>(half_patch * half_patch - y * y)))
				 );
    }

    for (int y = half_patch, v0 = 0; y >= vmin; --y) {
      while (umax[v0] == umax[v0 + 1]) {
	++v0;
      }
      umax[y] = v0;
      ++v0;
    }

    CHECK(cudaMemcpyToSymbol(d_umax,
                             umax.data(),
                             sizeof(int) * static_cast<size_t>(half_patch + 2),
                             0,
                             cudaMemcpyHostToDevice));
  }

  void initializeOrbPattern(const int patch_size, const int wta_k){
    static const int kBitPattern31[256 * 4] = {
      8,-3, 9,5/*mean (0), correlation (0)*/,
      4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
      -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
      7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
      2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
      1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
      -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
      -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
      -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
      10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
      -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
      -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
      7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
      -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
      -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
      -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
      12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
      -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
      -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
      11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
      4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
      5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
      3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
      -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
      -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
      -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
      -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
      -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
      -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
      5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
      5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
      1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
      9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
      4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
      2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
      -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
      -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
      4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
      0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
      -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
      -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
      -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
      8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
      0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
      7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
      -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
      10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
      -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
      10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
      -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
      -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
      3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
      5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
      -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
      3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
      2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
      -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
      -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
      -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
      -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
      6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
      -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
      -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
      -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
      3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
      -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
      -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
      2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
      -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
      -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
      5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
      -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
      -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
      -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
      10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
      7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
      -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
      -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
      7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
      -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
      -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
      -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
      7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
      -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
      1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
      2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
      -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
      -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
      7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
      1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
      9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
      -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
      -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
      7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
      12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
      6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
      5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
      2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
      3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
      2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
      9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
      -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
      -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
      1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
      6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
      2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
      6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
      3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
      7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
      -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
      -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
      -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
      -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
      8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
      4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
      -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
      4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
      -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
      -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
      7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
      -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
      -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
      8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
      -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
      1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
      7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
      -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
      11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
      -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
      3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
      5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
      0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
      -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
      0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
      -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
      5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
      3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
      -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
      -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
      -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
      6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
      -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
      -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
      1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
      4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
      -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
      2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
      -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
      4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
      -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
      -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
      7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
      4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
      -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
      7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
      7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
      -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
      -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
      -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
      2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
      10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
      -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
      8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
      2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
      -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
      -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
      -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
      5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
      -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
      -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
      -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
      -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
      -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
      2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
      -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
      -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
      -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
      -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
      6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
      -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
      11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
      7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
      -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
      -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
      -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
      -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
      -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
      -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
      -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
      -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
      1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
      1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
      9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
      5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
      -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
      -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
      -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
      -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
      8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
      2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
      7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
      -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
      -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
      4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
      3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
      -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
      5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
      4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
      -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
      0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
      -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
      3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
      -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
      8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
      -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
      2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
      10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
      6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
      -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
      -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
      -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
      -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
      -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
      4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
      2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
      6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
      3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
      11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
      -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
      4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
      2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
      -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
      -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
      -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
      6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
      0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
      -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
      -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
      -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
      5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
      2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
      -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
      9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
      11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
      3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
      -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
      3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
      -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
      5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
      8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
      7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
      -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
      7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
      9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
      7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
      -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
    };

    const int npoints = 512;
    int2 patternbuf[npoints];
    const int2* pattern0 = (const int2*)kBitPattern31;
    if (patch_size != 31)
      {
	pattern0 = patternbuf;
	// we always start with a fixed seed, to make patterns the same on each run
	srand(0x34985739);
	for (int i = 0; i < npoints; i++)
	  {
	    patternbuf[i].x = rand() % patch_size - -patch_size / 2;
	    patternbuf[i].y = rand() % patch_size - -patch_size / 2;
	  }
      }

    if (wta_k == 2)
      {
	//pattern = new int2[npoints];
	//memcpy(pattern, pattern0, npoints * sizeof(int2));
	CHECK(cudaMemcpyToSymbol(d_pattern, pattern0, npoints * sizeof(int2), 0, cudaMemcpyHostToDevice));
      }
    else
      {
	//initializeOrbPattern(pattern0, pattern, ntuples, wta_k, npoints);
	srand(0x12345678);
	int i, k, k1;
	int ntuples = 32 * 4;
	int2* pattern = new int2[ntuples * wta_k];
	for (i = 0; i < ntuples; i++)
	  {
	    for (k = 0; k < wta_k; k++)
	      {
		while (true)
		  {
		    int idx = rand() % npoints;
		    int2 pt = pattern0[idx];
		    for (k1 = 0; k1 < k; k1++)
		      {
			int2 pt1 = pattern[wta_k * i + k1];
			if (pt.x == pt1.x && pt.y == pt1.y)
			  break;
		      }
		    if (k1 == k)
		      {
			pattern[wta_k * i + k] = pt;
			break;
		      }
		  }
	      }
	  }

	CHECK(cudaMemcpyToSymbol(d_pattern, pattern, ntuples * wta_k * sizeof(int2), 0, cudaMemcpyHostToDevice));
	delete[] pattern; pattern = nullptr;
      }
  }


  void setGaussianKernel(){
    const float sigma = 2.f;
    const float inv_two_sigma_sq = -0.5f / (sigma * sigma);

    float kernel[GR + 1];
    float sum = 0.0f;

    for (int i = 0; i <= GR; ++i)
      {
        const float value = expf(static_cast<float>(i * i) * inv_two_sigma_sq);
        kernel[i] = value;
        sum += (i == 0 ? value : 2.0f * value);
      }

    const float normalize = 1.0f / sum;
    for (int i = 0; i <= GR; ++i)
      {
        kernel[i] *= normalize;
      }

    CHECK(cudaMemcpyToSymbol(d_gauss, kernel, (GR + 1) * sizeof(float), 0, cudaMemcpyHostToDevice));
  }


  void setScaleSqSq(){
    constexpr float max_intensity = 255.0f;
    constexpr float harris_size_f = static_cast<float>(HARRIS_SIZE);

    const float base_scale = 1.0f / (4.0f * harris_size_f * max_intensity);
    const float base_scale_sq = base_scale * base_scale;
    const float scale_sq_sq = base_scale_sq * base_scale_sq;

    CHECK(cudaMemcpyToSymbol(d_scale_sq_sq, &scale_sq_sq, sizeof(float), 0, cudaMemcpyHostToDevice));
  }


  __global__ void DownsampleKernel(const unsigned char* __restrict__ src,
                                 unsigned char* __restrict__ dst,
                                 int downsample_log2,
                                 int dst_width,
                                 int dst_height,
                                 int dst_pitch,
                                 int src_pitch){
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= dst_height) return;

    const int dst_row_offset = y * dst_pitch;
    const int src_row_offset = y * src_pitch;

    int x = blockIdx.x * blockDim.x * 4 + threadIdx.x;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (x >= dst_width) break;

        const int dst_idx = dst_row_offset + x;
        const int src_idx = (src_row_offset + x) << downsample_log2;

        dst[dst_idx] = src[src_idx];
        x += blockDim.x;
    }
}


  void makeOffsets(int* strides, int num_octaves){
    constexpr int kPatternSize = 16;
    constexpr int offsets[kPatternSize][2] = {
      {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3,  0}, { 3, -1}, { 2, -2}, { 1, -3},
      {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3,  0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    constexpr int kOffsetsPerOctave = 25;
    std::vector<int> host_offsets(static_cast<size_t>(kOffsetsPerOctave) * num_octaves);

    for (int octave = 0; octave < num_octaves; ++octave) {
      int* dst = host_offsets.data() + octave * kOffsetsPerOctave;
      const int stride = strides[octave];

      int k = 0;
      for (; k < kPatternSize; ++k) {
	dst[k] = offsets[k][0] + offsets[k][1] * stride;
      }
      for (; k < kOffsetsPerOctave; ++k) {
	dst[k] = dst[k - kPatternSize];
      }
    }

    CHECK(cudaMemcpyToSymbol(
			     dpixel,
			     host_offsets.data(),
			     host_offsets.size() * sizeof(int),
			     0,
			     cudaMemcpyHostToDevice
			     ));
  }



__device__ __forceinline__ 
bool has_contiguous_ones_on_circle(uint16_t pattern) {
    constexpr unsigned int kMask = (1u << kContiguousRequired) - 1u;

    unsigned int extended =
        (static_cast<unsigned int>(pattern) << kCirclePoints) |
         static_cast<unsigned int>(pattern);

    #pragma unroll
    for (int i = 0; i < kCirclePoints; ++i) {
        if ((extended & kMask) == kMask) {
            return true;
        }
        extended >>= 1;
    }
    return false;
}

__device__ __forceinline__ 
bool is_corner_from_mask(unsigned int compare) {
  const uint16_t high_pattern = static_cast<uint16_t>((compare >> 16) & 0xFFFFu);
  const uint16_t low_pattern  = static_cast<uint16_t>( compare        & 0xFFFFu);

  return has_contiguous_ones_on_circle(high_pattern) ||
    has_contiguous_ones_on_circle(low_pattern);
}



__device__ __forceinline__
void fastComparePixel(
		      const unsigned char* __restrict__ shared_img,
		      int x,                               
		      const unsigned char* __restrict__ row_lut,
		      int row_index,                       
		      int low_threshold,                   
		      int high_threshold,                  
		      unsigned int dark_bit,               
		      unsigned int bright_bit,             
		      unsigned int& mask                   
		      ){
  const int idx = row_lut[row_index] * FAST_WIDTH + x;
  const int val = static_cast<int>(shared_img[idx]);

  if (val < low_threshold) {
    mask |= dark_bit;
  } else if (val > high_threshold) {
    mask |= bright_bit;
  }
}

__device__ __forceinline__
unsigned int computeFast16PixelMask(
				    const unsigned char* __restrict__ shared_img,  
				    int center_x,                                  
				    const unsigned char* __restrict__ row_lut,              
				    int intensity_threshold                       
				    ){
  // Center pixel intensity
  const int center_idx   = row_lut[4] * FAST_WIDTH + center_x;
  const int center_val   = static_cast<int>(shared_img[center_idx]);
  const int low_threshold  = center_val - intensity_threshold;
  const int high_threshold = center_val + intensity_threshold;

  unsigned int mask = 0u;

  // index[1]: (0, 0), (-1, 0), (+1, 0)
  fastComparePixel(shared_img, center_x + 0, row_lut, 1, low_threshold, high_threshold, 0x80000000u, 0x00008000u, mask);

  fastComparePixel(shared_img, center_x - 1, row_lut, 1, low_threshold, high_threshold, 0x00010000u, 0x00000001u, mask);

  fastComparePixel(shared_img, center_x + 1, row_lut, 1, low_threshold, high_threshold, 0x40000000u, 0x00004000u, mask);

  // index[2]: (-2, ·), (+2, ·)
  fastComparePixel(shared_img, center_x - 2, row_lut, 2, low_threshold, high_threshold, 0x00020000u, 0x00000002u, mask);

  fastComparePixel(shared_img, center_x + 2, row_lut, 2, low_threshold, high_threshold, 0x20000000u, 0x00002000u, mask);

  // index[3]: (-3, ·), (+3, ·)
  fastComparePixel(shared_img, center_x - 3, row_lut, 3, low_threshold, high_threshold, 0x00040000u, 0x00000004u, mask);

  fastComparePixel(shared_img, center_x + 3, row_lut, 3, low_threshold, high_threshold, 0x10000000u, 0x00001000u, mask);

  // index[4]: (-3, ·), (+3, ·)
  fastComparePixel(shared_img, center_x - 3, row_lut, 4, low_threshold, high_threshold, 0x00080000u, 0x00000008u, mask);

  fastComparePixel(shared_img, center_x + 3, row_lut, 4, low_threshold, high_threshold, 0x08000000u, 0x00000800u, mask);

  // index[5]: (-3, ·), (+3, ·)
  fastComparePixel(shared_img, center_x - 3, row_lut, 5, low_threshold, high_threshold, 0x00100000u, 0x00000010u, mask);

  fastComparePixel(shared_img, center_x + 3, row_lut, 5, low_threshold, high_threshold, 0x04000000u, 0x00000400u, mask);

  // index[6]: (-2, ·), (+2, ·)
  fastComparePixel(shared_img, center_x - 2, row_lut, 6, low_threshold, high_threshold, 0x00200000u, 0x00000020u, mask);

  fastComparePixel(shared_img, center_x + 2, row_lut, 6, low_threshold, high_threshold, 0x02000000u, 0x00000200u, mask);

  // index[7]: (-1, ·), (+1, ·), (0, ·)
  fastComparePixel(shared_img, center_x - 1, row_lut, 7, low_threshold, high_threshold, 0x00400000u, 0x00000040u, mask);

  fastComparePixel(shared_img, center_x + 1, row_lut, 7, low_threshold, high_threshold, 0x00800000u, 0x00000080u, mask);

  fastComparePixel(shared_img, center_x + 0, row_lut, 7, low_threshold, high_threshold, 0x01000000u, 0x00000100u, mask);

  return mask;
}



  __device__ __forceinline__
  void computeHarrisResponse(
					const int*  __restrict__ dxx,         
					const int*  __restrict__ dyy,         
					const int*  __restrict__ dxy,         
					int                     ix,           
					int                     iy,           
					int                     octave,       
					int                     img_width,    
					int                     row_cells,    
					int                     row_stride,   
					int                     tile_offset,  
					float*       __restrict__ score_map,  
					int*         __restrict__ octave_map, 
					bool                    is_initial    
					){
    int sum_dxx = 0;
    int sum_dyy = 0;
    int sum_dxy = 0;

#pragma unroll
    for (int j = 0; j < HARRIS_SIZE; ++j) {
      const int idx = threadIdx.x + j;
      sum_dxx += dxx[idx];
      sum_dyy += dyy[idx];
      sum_dxy += dxy[idx];
    }

    const int trace = sum_dxx + sum_dyy;
    const float det = static_cast<float>(sum_dxx) * static_cast<float>(sum_dyy)
      - static_cast<float>(sum_dxy) * static_cast<float>(sum_dxy);

    const float score =
      (det - HARRIS_K * static_cast<float>(trace * trace)) * d_scale_sq_sq;

    const int x_base = ix - 1;
    const int y_base = 4 + tile_offset + iy * row_cells;

    const int x = x_base << octave;
    const int y = y_base << octave;
    const int map_idx = y * row_stride + x;

    const int right_border = is_initial ? 3 : 4;
    const int max_x_base   = img_width - right_border;

    if (x_base < max_x_base && score_map[map_idx] < score) {
      score_map[map_idx]  = score;
      octave_map[map_idx] = octave;
    }
  }



  __device__ __forceinline__
  void updateFastSharedTile(
				       const unsigned char* __restrict__ d_image,
				       unsigned char*       __restrict__ s_tile, 				    	int                  image_height,
				       int                  image_pitch,
				       int                  row_offset,
				       unsigned char*       __restrict__ row_index,					bool                 is_initial_pass){
    constexpr int kCoreWidth    = X_FAST;       
    constexpr int kTileWidth    = FAST_WIDTH;   
    constexpr int kTileHeight   = FAST_HEIGHT;  
    constexpr int kBorderRadius = 4;            
    constexpr int kHaloWidth    = 2 * kBorderRadius;
    constexpr int kRingPos      = 8;            
    constexpr int kLookahead    = 9;            

    const int local_x   = threadIdx.x;
    const int block_x   = blockIdx.x;
    const int block_y   = blockIdx.y;

    const int stripe_height = image_height / gridDim.y;
    const int base_row      = block_y * stripe_height;

    const int global_x = block_x * kCoreWidth + local_x + kBorderRadius;
    const int shared_x = local_x + kBorderRadius;  

    if (global_x < kBorderRadius || global_x >= image_pitch - kBorderRadius) {
      return;
    }

    if (is_initial_pass){
#pragma unroll
        for (int row = 0; row < kTileHeight; ++row){
            const int img_y = base_row + row;
            const int img_row_offset = img_y * image_pitch;

            s_tile[(shared_x - kBorderRadius) + row * kTileWidth] =
	      d_image[img_row_offset + global_x - kBorderRadius];
	    
	    if (local_x > kCoreWidth - (kHaloWidth + 1)){
                s_tile[(shared_x + kBorderRadius) + row * kTileWidth] =
		  d_image[img_row_offset + global_x + kBorderRadius];
	      }
	  }
      }
    else{
        const int tile_row = static_cast<int>(row_index[kRingPos]); 
        const int img_y    = base_row + row_offset + kLookahead;
        const int img_row_offset = img_y * image_pitch;

        s_tile[local_x + tile_row * kTileWidth] =
	  d_image[img_row_offset + global_x - kBorderRadius];

        if (local_x > kCoreWidth - (kHaloWidth + 1)){
            s_tile[(local_x + kHaloWidth) + tile_row * kTileWidth] =
	      d_image[img_row_offset + global_x + kBorderRadius];
	  }
      }
  }


  __device__ __forceinline__
  void computeSobelBlock(unsigned char* smem,
			 unsigned char* grad_x,
			 unsigned char* grad_y,
			 int* sum_dxx,
			 int* sum_dyy,
			 int* sum_dxy,
			 const unsigned char* row_index)
  {
    const int lane      = threadIdx.x;
    const int center    = lane + 4;
    const bool has_pair = (lane < HARRIS_SIZE - 1);

    int dxx_primary  = 0;
    int dyy_primary  = 0;
    int dxy_primary  = 0;
    int dxx_secondary = 0;
    int dyy_secondary = 0;
    int dxy_secondary = 0;

    for (int i = 0; i < FAST_HEIGHT; ++i) {
      const int row_offset   = static_cast<int>(row_index[i]) * FAST_WIDTH;
      const int output_base  = i * FAST_WIDTH + lane;

      const int left_center  = row_offset + center;
      const int gx_left      = static_cast<int>(smem[left_center - 2]) - static_cast<int>(smem[left_center - 4]);
      const int gy_left      = static_cast<int>(smem[left_center - 2])
	+ static_cast<int>(2) * static_cast<int>(smem[left_center - 3])
	+ static_cast<int>(smem[left_center - 4]);

      grad_x[output_base] = gx_left;
      grad_y[output_base] = gy_left;

      if (has_pair) {
	const int output_base_pair = i * FAST_WIDTH + lane + X_FAST;
	const int right_base       = row_offset + lane + X_FAST;

	const int gx_right = static_cast<int>(smem[right_base + 2]) - static_cast<int>(smem[right_base]);
	const int gy_right = static_cast<int>(smem[right_base])
	  + static_cast<int>(2) * static_cast<int>(smem[right_base + 1])
	  + static_cast<int>(smem[right_base + 2]);

	grad_x[output_base_pair] = gx_right;
	grad_y[output_base_pair] = gy_right;
      }
    }

    __syncthreads();

    for (int i = 0; i < HARRIS_SIZE; ++i) {
      const int row0      = i * FAST_WIDTH + lane;
      const int row1      = (i + 1) * FAST_WIDTH + lane;
      const int row2      = (i + 2) * FAST_WIDTH + lane;

      const int gx0       = grad_x[row0] + 2 * grad_x[row1] + grad_x[row2];
      const int gy0       = grad_y[row2] - grad_y[row0];

      dxx_primary        += gx0 * gx0;
      dyy_primary        += gy0 * gy0;
      dxy_primary        += gx0 * gy0;

      if (has_pair) {
	const int col_offset_pair = lane + X_FAST;

	const int row0_pair = i * FAST_WIDTH + col_offset_pair;
	const int row1_pair = (i + 1) * FAST_WIDTH + col_offset_pair;
	const int row2_pair = (i + 2) * FAST_WIDTH + col_offset_pair;

	const int gx1 = grad_x[row0_pair] + 2 * grad_x[row1_pair] + grad_x[row2_pair];
	const int gy1 = grad_y[row2_pair] - grad_y[row0_pair];

	dxx_secondary += gx1 * gx1;
	dyy_secondary += gy1 * gy1;
	dxy_secondary += gx1 * gy1;
      }
    }

    sum_dxx[lane] = dxx_primary;
    sum_dyy[lane] = dyy_primary;
    sum_dxy[lane] = dxy_primary;

    if (has_pair) {
      const int lane_pair = lane + X_FAST;
      sum_dxx[lane_pair]  = dxx_secondary;
      sum_dyy[lane_pair]  = dyy_secondary;
      sum_dxy[lane_pair]  = dxy_secondary;
    }
  }


  __global__ void computeExtremaMapKernel(
					  const unsigned char* __restrict__ image,
					  float* __restrict__ response_map,
					  int* __restrict__ layer_map,
					  int corner_threshold,
					  int octave,
					  int image_width,
					  int image_height,
					  int image_pitch,
					  int layer_offset)
  {
    __shared__ unsigned char shared_image[FAST_WIDTH * FAST_HEIGHT];
    __shared__ unsigned char shared_gx[FAST_WIDTH * FAST_HEIGHT];
    __shared__ unsigned char shared_gy[FAST_WIDTH * FAST_HEIGHT];
    __shared__ int shared_dxx[FAST_WIDTH - 2];
    __shared__ int shared_dyy[FAST_WIDTH - 2];
    __shared__ int shared_dxy[FAST_WIDTH - 2];

    const int local_x  = threadIdx.x + 4;
    const int global_x = blockIdx.x * X_FAST + threadIdx.x + 4;

    const int rows_per_group = image_height / gridDim.y;
    const int group_id_y     = blockIdx.y;

    unsigned char ring_index[9];
    for (int i = 0; i < 8; ++i) {
      ring_index[i] = static_cast<unsigned char>(i & 0x7);
    }
    ring_index[8] = 8;

    updateFastSharedTile(image, shared_image, image_height, image_pitch, 0, ring_index, true);
    __syncthreads();

    unsigned int fast_mask = computeFast16PixelMask(shared_image, local_x, ring_index, corner_threshold);
    bool is_corner = is_corner_from_mask(fast_mask);

    bool compute_sobel = __any_sync(0xFFFFFFFF, is_corner);

    if (compute_sobel) {
      computeSobelBlock(shared_image, shared_gx, shared_gy, shared_dxx, shared_dyy, shared_dxy, ring_index);
      if (is_corner) {
	computeHarrisResponse(
			      shared_dxx,
			      shared_dyy,
			      shared_dxy,
			      global_x,
			      group_id_y,
			      octave,
			      image_width,
			      rows_per_group,
			      layer_offset,
			      0,
			      response_map,
			      layer_map,
			      true);
      }
    }

    for (int tile_offset = 0; tile_offset < rows_per_group - 1; ++tile_offset) {
      const int row_index = tile_offset + 9 + group_id_y * rows_per_group;
      if (row_index < image_height) {
	for (int i = 0; i < 8; ++i) {
	  ring_index[i] = static_cast<unsigned char>((tile_offset + i + 1) & 0x7);
	}
	ring_index[8] = static_cast<unsigned char>((tile_offset + 9) % 9);

	updateFastSharedTile(image, shared_image, image_height, image_pitch, tile_offset, ring_index, false);
	__syncthreads();

	fast_mask = computeFast16PixelMask(shared_image, local_x, ring_index, corner_threshold);
	bool is_corner_tile = is_corner_from_mask(fast_mask);
	compute_sobel = __any_sync(0xFFFFFFFF, is_corner_tile);

	if (compute_sobel) {
	  computeSobelBlock(shared_image, shared_gx, shared_gy, shared_dxx, shared_dyy, shared_dxy, ring_index);
	  if (is_corner_tile) {
	    computeHarrisResponse(
				  shared_dxx,
				  shared_dyy,
				  shared_dxy,
				  global_x,
				  group_id_y,
				  octave,
				  image_width,
				  rows_per_group,
				  layer_offset,
				  tile_offset + 1,
				  response_map,
				  layer_map,
				  false);
	  }
	}
      }
    }
  }



  __global__ void orbNonMaxSuppressionKernel(
					     OrbPoint* __restrict__ points,
					     const float* __restrict__ score_map,
					     const int* __restrict__ octave_map,
					     int border,
					     int image_width,
					     int image_height,
					     int stride){
    int y = blockIdx.y * blockDim.y + threadIdx.y + border;
    if (y >= image_height - border) {
      return;
    }

    int row_offset = y * stride;
    int x = blockIdx.x * (blockDim.x * 4) + threadIdx.x + border;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (x >= image_width - border) {
	return;
      }

      int idx = row_offset + x;
      float center_score = score_map[idx];

      if (center_score > 0.0f && d_point_counter < d_max_num_points) {
	int octave = octave_map[idx];
	int radius = 1 << octave;

	bool suppressed = false;
	int window_row_offset = (y - radius) * stride;

	for (int dy = -radius; dy <= radius && !suppressed; ++dy) {
	  int neighbor_idx = window_row_offset + x - radius;

	  for (int dx = -radius; dx <= radius; ++dx) {
	    if (dx != 0 || dy != 0) {
	      float neighbor_score = score_map[neighbor_idx];
	      if (neighbor_score > 0.0f &&
		  (neighbor_score > center_score ||
		   (neighbor_score == center_score && dx <= 0 && dy <= 0))) {
		suppressed = true;
		break;
	      }
	    }
	    ++neighbor_idx;
	  }

	  window_row_offset += stride;
	}

	if (!suppressed && d_point_counter < d_max_num_points) {
	  unsigned int point_index = atomicInc(&d_point_counter, 0x7fffffff);
	  if (point_index < d_max_num_points) {
	    points[point_index].x      = x;
	    points[point_index].y      = y;
	    points[point_index].octave = octave;
	    points[point_index].score  = center_score;
	  }
	}
      }

      x += blockDim.x;
    }
  }




  void detectFastKeypointsWithNms(
				  unsigned char* image_device,
				  unsigned char* pyramid_device,
				  float* score_buffer,
				  OrbData& keypoints,
				  int* pyramid_meta,
				  int num_octaves,
				  int fast_threshold,
				  int border,
				  bool use_harris_score){
    if (border < 3) {
      border = 3;
    }

    int* octave_sizes = pyramid_meta;
    int* widths       = octave_sizes + num_octaves;
    int* heights      = widths       + num_octaves;
    int* pitches      = heights      + num_octaves;
    int* offsets      = pitches      + num_octaves;

    float* score_map = score_buffer;
    int* layer_map   = reinterpret_cast<int*>(score_map + octave_sizes[0]);

    dim3 block_pyr(X2, X2);
    dim3 grid_pyr;

    CHECK(cudaMemcpy(pyramid_device,
                     image_device,
                     octave_sizes[0] * sizeof(unsigned char),
                     cudaMemcpyDeviceToDevice));

    int scale_factor = 1;
    for (int octave = 1; octave < num_octaves; ++octave) {
      grid_pyr.x = (widths[octave]  + X2 * 4 - 1) / (X2 * 4);
      grid_pyr.y = (heights[octave] + X2     - 1) /  X2;

      DownsampleKernel<<<grid_pyr, block_pyr>>>(
						image_device,
						pyramid_device + offsets[octave],
						scale_factor,
						widths[octave],
						heights[octave],
						pitches[octave],
						pitches[0]);

      CHECK(cudaDeviceSynchronize());
      ++scale_factor;
    }

    dim3 block_fast(X_FAST, 1);
    dim3 grid_detect;

    for (int octave = 0; octave < num_octaves; ++octave) {
      grid_detect.x = (widths[octave] - 8 + X_FAST - 1) / X_FAST;
      grid_detect.y = (octave < 2) ? (heights[octave] / 2)
	: (heights[octave] / 2 - 1);

      computeExtremaMapKernel<<<grid_detect, block_fast>>>(
							   pyramid_device + offsets[octave],
							   score_map,
							   layer_map,
							   fast_threshold,
							   octave,
							   widths[octave],
							   heights[octave],
							   pitches[octave],
							   pitches[0]);

      CHECK(cudaDeviceSynchronize());
    }

    int total_border = border * 2;

    dim3 grid_nms(
		  (widths[0]  - total_border + X2 * 4 - 1) / (X2 * 4),
		  (heights[0] - total_border + X2     - 1) /  X2
		  );

    orbNonMaxSuppressionKernel<<<grid_nms, block_pyr>>>(
							keypoints.d_data,
							score_map,
							layer_map,
							border,
							widths[0],
							heights[0],
							pitches[0]);

    CHECK(cudaDeviceSynchronize());
    CheckMsg("detectFastKeypointsWithNms() execution failed!\n");
  }
}
