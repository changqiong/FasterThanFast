#pragma once
#include "structures.h"
#include "utils.h"
#include <vector>



namespace orb
{

	class Orbor
	{
	public:
		Orbor();
		~Orbor();

		/* Initialization */
		void init(int _noctaves = 5, int _edge_threshold = 31, int _wta_k = 2, ScoreType _score_type = ScoreType::HARRIS_SCORE,
			int _patch_size = 31, int _fast_threshold = 20, int _retain_topn = -1, int _max_pts = 1000);

		/* Detect keypoints and compute descriptors */
		void detectAndCompute(unsigned char* image, OrbData& result, int3 whp, void** desc_addr = NULL, const bool compute_desc = true);

		/* Initialize data */
		void initOrbData(OrbData& data, const int max_pts, const bool host, const bool dev);

		/* Free data */
		void freeOrbData(OrbData& data);

	private:
		//static const int kbytes = 32;
		// The number of pyramid levels.		
		int noctaves = 5;												//金字塔层数
		// Truly octave layers
		int max_octave = 5;												
		// This is size of the border where the features are not detected.
		int edge_threshold = 31;											//边缘
		// The number of points that produce each element of the oriented BRIEF descriptor.
		int wta_k = 2;												//生成定向BRIEF描述符的每个元素的点数。
		// The default HARRIS_SCORE means that Harris algorithm is used to rank features
		// (the score is written to KeyPoint::scoreand is used to retain best nfeatures features);
		// FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
		// but it is a little faster to compute.
		ScoreType score_type = ScoreType::HARRIS_SCORE;
		// The size of the patch used by the oriented BRIEF descriptor.							//定向BRIEF描述符使用的patch大小
		int patch_size = 31;

		// The fast threshold
		int fast_threshold = 20;

		// The top-n best keypoints retained for matching. If less and equal to 0, retain all
		int retain_topn = -1;		
										//保留前n个最佳关键点进行匹配。如果小于等于0，则保留所有
		// Max number of points
		int max_pts = 1000;

		// Size of image for reusing
		int width = -1;
		int height = -1;

		// Memory for reusing
		unsigned char* omem = NULL;										// Shared memory of scaled images
		float* vmem = NULL;
															// Shared memory of score maps for non-max suppression
		// Size of memory
		size_t obytes = 0;
		size_t vbytes = 0;

		// Size of octave layers
		std::vector<int> oszp;

		// Point counter
		unsigned int* d_point_counter_addr = NULL;

		/* Update parameters related to memory */
		void updateParam(int3 whp0);

		/* Detect keypoints */
		void detectKeypoints(unsigned char* d_image, OrbData& keypoints);

	};

}

