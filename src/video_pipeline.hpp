// video_pipeline.hpp — Video processing pipeline for background effects.
//
// Wraps OpenVINO inference for MODNet (background segmentation).
// Provides frame capture from webcam, model inference, blur composite,
// and output to v4l2loopback virtual camera.
//
// Phase 3: MODNet background segmentation + blur composite.
// Phase 4: face-detection-adas-0001 for auto-framing overlay.
//
#pragma once

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

// ─── Video constants ────────────────────────────────────────────────────────

static constexpr int VIDEO_INPUT_WIDTH  = 640;
static constexpr int VIDEO_INPUT_HEIGHT = 480;
static constexpr int VIDEO_OUTPUT_WIDTH  = 640;
static constexpr int VIDEO_OUTPUT_HEIGHT = 480;
static constexpr int VIDEO_FPS           = 25;

// MODNet model input (square crop, resized)
static constexpr int MODNET_INPUT_SIZE   = 256;

// ─── MODNet background segmentation engine ─────────────────────────────────

class ModNetEngine {
public:
    ModNetEngine() = default;
    ~ModNetEngine() = default;

    bool init(const std::string& model_xml, const std::string& device = "NPU") {
        try {
            ov::Core core("modnet-engine");
            model_ = std::make_unique<ov::CompiledModel>(
                core.compile_model(model_xml, device));

            // Static shape: [1, 3, 256, 256]
            // NPU requires this to be fixed at compile time (already baked into IR)

            fprintf(stderr, "[modnet] loaded model (%s)\n", device.c_str());
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "[modnet] ERROR: %s\n", e.what());
            return false;
        }
    }

    // Process one frame: returns alpha matte [0.0, 1.0]
    cv::Mat process(const cv::Mat& frame_bgr) {
        if (!model_) {
            return fallback_matte(frame_bgr);
        }

        // Preprocess: resize to 256×256, normalize, BGR→RGB, NCHW
        cv::Mat resized;
        cv::resize(frame_bgr, resized, cv::Size(MODNET_INPUT_SIZE, MODNET_INPUT_SIZE));

        resized.convertTo(resized, CV_32FC3, 1.0/255.0);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);

        // Create inference request
        ov::InferRequest req = model_->create_infer_request();
        auto input_tensor = req.get_tensor("input");  // name TBD

        // Copy channel data
        size_t plane_size = MODNET_INPUT_SIZE * MODNET_INPUT_SIZE * sizeof(float);
        for (int c = 0; c < 3; c++) {
            memcpy(input_tensor.data<float>() + c * plane_size,
                   channels[c].data, plane_size);
        }

        req.infer();

        // Output tensor
        auto output_tensor = req.get_tensor("output");
        cv::Mat matte_small(MODNET_INPUT_SIZE, MODNET_INPUT_SIZE, CV_32FC1,
                            output_tensor.data<float>());

        cv::Mat matte;
        cv::resize(matte_small, matte, cv::Size(frame_bgr.cols, frame_bgr.rows));

        return matte;
    }

    bool is_loaded() const { return model_ != nullptr; }

private:
    // Fallback: simple skin-tone or color-based matte (no model)
    cv::Mat fallback_matte(const cv::Mat& frame) {
        cv::Mat matte(frame.rows, frame.cols, CV_32FC1, 0.5f);
        return matte;
    }

    std::unique_ptr<ov::CompiledModel> model_;
};

// ─── Full video pipeline: webcam → MODNet → blur composite → v4l2loopback ──

class VideoPipeline {
public:
    VideoPipeline() = default;

    struct Config {
        int input_width  = VIDEO_INPUT_WIDTH;
        int input_height = VIDEO_INPUT_HEIGHT;
        int output_width  = VIDEO_OUTPUT_WIDTH;
        int output_height = VIDEO_OUTPUT_HEIGHT;
        int fps           = VIDEO_FPS;
        float blur_strength = 0.5f;  // 0.0–1.0
        std::string bg_image_path = "";  // optional: background image instead of blur
        std::string model_dir = "";
    };

    bool init(const Config& cfg) {
        cfg_ = cfg;

        // Init MODNet
        if (!cfg.model_dir.empty()) {
            engine_ = std::make_unique<ModNetEngine>();
            if (!engine_->init(cfg.model_dir + "/model.xml", "NPU")) {
                // Fallback to GPU
                if (!engine_->init(cfg.model_dir + "/model.xml", "GPU")) {
                    fprintf(stderr, "[video] MODNet not available, using fallback\n");
                }
            }
        }

        // Load background image (if specified)
        if (!cfg.bg_image_path.empty()) {
            bg_image_ = cv::imread(cfg.bg_image_path);
            if (bg_image_.empty()) {
                fprintf(stderr, "[video] WARNING: could not load bg image %s\n",
                        cfg.bg_image_path.c_str());
            } else {
                cv::resize(bg_image_, bg_image_,
                           cv::Size(cfg.output_width, cfg.output_height));
            }
        }

        fprintf(stderr, "[video] pipeline ready (%dx%d @ %dfps, blur=%.2f)\n",
                cfg_.output_width, cfg_.output_height, cfg_.fps, cfg_.blur_strength);
        return true;
    }

    // Process one frame
    cv::Mat process_frame(const cv::Mat& frame_bgr) {
        if (frame_bgr.empty()) return frame_bgr;

        cv::Mat resized_frame;
        cv::resize(frame_bgr, resized_frame,
                   cv::Size(cfg_.output_width, cfg_.output_height));

        if (!engine_ || !engine_->is_loaded()) {
            return resized_frame;  // passthrough
        }

        // Get alpha matte
        cv::Mat alpha = engine_->process(resized_frame);

        // Create output: foreground + blurred background
        cv::Mat output = resized_frame.clone();

        if (!bg_image_.empty()) {
            // Use custom background image
            bg_image_.copyTo(output, 1.0f - alpha);
        } else {
            // Blur: downscale → upscale for fast blur
            float blur_ratio = 0.1f + cfg_.blur_strength * 0.2f;  // 0.1–0.3
            int blur_w = std::max(1, (int)(cfg_.output_width * blur_ratio));
            int blur_h = std::max(1, (int)(cfg_.output_height * blur_ratio));

            cv::Mat blurred;
            cv::resize(resized_frame, blurred, cv::Size(blur_w, blur_h));
            cv::resize(blurred, blurred, cv::Size(cfg_.output_width, cfg_.output_height));

            // Composite: foreground (alpha) + blurred background (1-alpha)
            for (int y = 0; y < output.rows; y++) {
                float* a = alpha.ptr<float>(y);
                cv::Vec3b* out = output.ptr<cv::Vec3b>(y);
                cv::Vec3b* bg  = blurred.ptr<cv::Vec3b>(y);
                for (int x = 0; x < output.cols; x++) {
                    float alpha_val = a[x];
                    float inv = 1.0f - alpha_val;
                    out[x][0] = (uchar)(out[x][0] * alpha_val + bg[x][0] * inv);
                    out[x][1] = (uchar)(out[x][1] * alpha_val + bg[x][1] * inv);
                    out[x][2] = (uchar)(out[x][2] * alpha_val + bg[x][2] * inv);
                }
            }
        }

        return output;
    }

    void set_blur_strength(float s) { cfg_.blur_strength = s; }
    void set_bg_image(const std::string& path) {
        bg_image_ = cv::imread(path);
        if (!bg_image_.empty() && cfg_.output_width > 0) {
            cv::resize(bg_image_, bg_image_,
                       cv::Size(cfg_.output_width, cfg_.output_height));
        }
    }

private:
    Config cfg_;
    std::unique_ptr<ModNetEngine> engine_;
    cv::Mat bg_image_;
};
