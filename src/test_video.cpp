// test_video.cpp — Test the video pipeline end-to-end.
// Captures a frame from webcam, processes through MODNet (or fallback),
// applies blur composite, writes output image.
//
// Usage: ./test_video --no-model  (tests fallback without downloading models)
//         ./test_video --model <dir>  (uses MODNet model)
//
#include "video_pipeline.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string model_dir = "";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-model") model_dir = "";
        if (std::string(argv[i]) == "--model" && i+1 < argc) model_dir = argv[++i];
    }

    // Load test frame
    cv::Mat frame = cv::imread("/tmp/test_frame_raw.png");
    if (frame.empty()) {
        std::cerr << "ERROR: could not load /tmp/test_frame_raw.png\n";
        std::cerr << "Run: ffmpeg -y -f v4l2 -i /dev/video0 -frames:v 1 /tmp/test_frame_raw.png\n";
        return 1;
    }

    std::cout << "Input frame: " << frame.cols << "x" << frame.rows
              << " channels=" << frame.channels() << "\n";

    // Init pipeline
    VideoPipeline::Config cfg;
    cfg.model_dir = model_dir;
    cfg.blur_strength = 0.5f;  // 50% blur
    cfg.output_width = 640;
    cfg.output_height = 480;

    VideoPipeline pipeline;
    if (!pipeline.init(cfg)) {
        std::cerr << "ERROR: pipeline init failed\n";
        return 1;
    }

    // Process frame
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat output = pipeline.process_frame(frame);
    auto t1 = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Processed in " << ms << "ms\n";
    std::cout << "Output: " << output.cols << "x" << output.rows << "\n";

    // Save result
    cv::imwrite("/tmp/test_frame_processed.png", output);
    std::cout << "Saved: /tmp/test_frame_processed.png\n";

    // Compare (simple: check pixels changed)
    cv::Mat raw_resized;
    cv::resize(frame, raw_resized, cv::Size(640, 480));
    int diff_count = 0;
    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {
            cv::Vec3b a = output.at<cv::Vec3b>(y, x);
            cv::Vec3b b = raw_resized.at<cv::Vec3b>(y, x);
            if (abs(a[0]-b[0]) > 10 || abs(a[1]-b[1]) > 10 || abs(a[2]-b[2]) > 10) {
                diff_count++;
            }
        }
    }
    float pct = 100.0f * diff_count / (output.rows * output.cols);
    std::cout << "Pixels changed: " << diff_count << " (" << pct << "%)\n";

    if (model_dir.empty()) {
        std::cout << "\nNOTE: No model loaded — using fallback (passthrough + resize).\n";
        std::cout << "Download MODNet: ./src/download_models.sh modnet\n";
    } else {
        std::cout << "\nMODNet background blur applied.\n";
    }

    return 0;
}
