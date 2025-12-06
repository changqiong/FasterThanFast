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


#include "orb.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <memory>

void drawKeypoints(orb::OrbData& a, cv::Mat& img, cv::Mat& dst);


int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Argument parsing
    // -------------------------------------------------------------------------
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [save_dir]\n";
        return EXIT_FAILURE;
    }

    const std::string input_image_path = argv[1];
    // -------------------------------------------------------------------------
    // Load image
    // -------------------------------------------------------------------------
    cv::Mat img = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_image_path << '\n';
        return EXIT_FAILURE;
    }

    const unsigned int width  = static_cast<unsigned int>(img.cols);
    const unsigned int height = static_cast<unsigned int>(img.rows);

    std::cout << "Image size = (" << width << ", " << height << ")\n";

    constexpr int   kNumOctaves        = 5;
    constexpr int   kEdgeThreshold     = 31;
    constexpr int   kWtaK              = 4;
    constexpr auto  kScoreType         = orb::ScoreType::HARRIS_SCORE;
    constexpr int   kPatchSize         = 31;
    constexpr int   kFastThreshold     = 20;
    constexpr int   kMaxKeypoints      = 10000;
    constexpr int   kRetainTopN        = 0;      // 0 = keep all up to kMaxKeypoints
    constexpr int   kNumRepeats        = 100;    // for timing average

    // -------------------------------------------------------------------------
    // GPU memory allocation and upload
    // -------------------------------------------------------------------------

    GpuTimer timer(0);
    unsigned char* d_image = nullptr;

    int3 imageSize{};
    imageSize.x = static_cast<int>(width);
    imageSize.y = static_cast<int>(height);

    const size_t image_bytes = sizeof(unsigned char) * imageSize.x * imageSize.y;

    CHECK(cudaMalloc(&d_image, image_bytes));
    CHECK(cudaMemcpy(d_image,
                     img.data,
                     image_bytes,
                     cudaMemcpyHostToDevice));

    const float t0 = timer.read();

    // -------------------------------------------------------------------------
    // Initialize ORB detector and device-side buffers
    // -------------------------------------------------------------------------
    auto FASTfeature = std::make_unique<orb::Orbor>();
    FASTfeature->init(kNumOctaves,
                   kEdgeThreshold,
                   kWtaK,
                   kScoreType,
                   kPatchSize,
                   kFastThreshold,
                   kRetainTopN,
                   kMaxKeypoints);

    orb::OrbData feature_data{};
    FASTfeature->initOrbData(feature_data, kMaxKeypoints, /*allocate_kpts=*/true, /*allocate_scores=*/true);

    unsigned char* d_orb_descriptors = nullptr;

    const float t1 = timer.read();

    // -------------------------------------------------------------------------
    // Detection & description (repeated for average timing)
    // -------------------------------------------------------------------------
    for (int i = 0; i < kNumRepeats; ++i) {
        FASTfeature->detectAndCompute(
            d_image,
            feature_data,
            imageSize,
            reinterpret_cast<void**>(&d_orb_descriptors),
            /*do_orientation=*/true
        );
    }

    const float t2 = timer.read();

    // -------------------------------------------------------------------------
    // Print statistics
    // -------------------------------------------------------------------------
    std::cout << "Number of features: " << feature_data.num_pts << '\n';

    std::cout << "Time for image upload:            " << t0              << " ms\n"
              << "Time for ORB init (once):         " << (t1 - t0)       << " ms\n"
              << "Time for detect+describe (avg):   " << (t2 - t1) / kNumRepeats << " ms\n";

    // -------------------------------------------------------------------------
    // Visualization & saving
    // -------------------------------------------------------------------------
    cv::Mat keypoint_vis;
    drawKeypoints(feature_data, img, keypoint_vis);

    const std::string out_path = "./result.png";
    cv::imwrite(out_path, keypoint_vis);
    std::cout << "Saved result to: " << out_path << '\n';

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    FASTfeature->freeOrbData(feature_data);

    if (d_orb_descriptors != nullptr) {
        CHECK(cudaFree(d_orb_descriptors));
    }

    CHECK(cudaFree(d_image));

    return EXIT_SUCCESS;
}


void drawKeypoints(orb::OrbData& a, cv::Mat& img, cv::Mat& dst)
{
  orb::OrbPoint* data = a.h_data;
  cv::merge(std::vector<cv::Mat>{ img, img, img }, dst);
  for (int i = 0; i < a.num_pts; i++)
    {
      orb::OrbPoint& p = data[i];
      cv::Point center(cvRound(p.x), cvRound(p.y));
      cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
      cv::circle(dst, center, MAX(1, MIN(5, log10(p.score))), color);
    }
}
