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

#pragma once
#include "structures.h"
#include "utils.h"


namespace orb
{

	/* Set the maximum number of keypoints. */
	void setMaxKeypointCount(const int num);

	/* Get the address of point counter */
	void getPointCounterDeviceAddress(void** addr);

	/* Compute FAST threshold LUT */
	void initializeFastThresholdLUT(int fast_threshold);

	/* Compute umax for angle computation */
	void initializeUmaxTable(const int patch_size);

	/* Set pattern for feature computation */
	void initializeOrbPattern(const int patch_size, const int wta_k);

	/* Set Gaussain kernel. Size 7, sigma 2 */
	void setGaussianKernel();

	/* Set scale factor for harris score. */
	void setScaleSqSq();

	/* Make offsets for FAST keypoints detection */
	void makeOffsets(int* pitchs, int noctaves);


	/* Find extreme by FAST */
	void detectFastKeypointsWithNms(unsigned char* image_device, unsigned char* pyramid_device, float* score_buffer, OrbData& keypoints, int* pyramid_meta, int num_octaves, int fast_threshold, int border, bool use_harris_score);
}
