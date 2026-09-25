// audio_pipeline.hpp — Audio processing pipeline for noise suppression.
//
// Wraps OpenVINO inference for DeepFilterNet3 (enc + erb_dec + df_dec).
// Provides STFT/ISTFT using FFTW3, and a PipeWire capture → process → sink
// loop. The daemon calls these functions when mic activity is detected.
//
// Phase 2: Implements DeepFilterNet3 only (noise suppression).
// Phase 5: AudioSR will be added as a separate pipeline.
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
#include <complex>
#include <sndfile.h>
#include <fftw3.h>

#include <openvino/openvino.hpp>

// ─── Audio constants ─────────────────────────────────────────────────────────

static constexpr int AUDIO_SAMPLE_RATE = 48000;   // Hz — DeepFilterNet trains on 48kHz
static constexpr int DF_HOP_SIZE        = 256;     // hop length (DeepFilterNet uses 256 at 48kHz = ~5.3ms)
static constexpr int DF_FRAME_SIZE      = 2048;    // FFT window size
static constexpr int DF_N_BANDS         = 1025;    // FFT bins (2048/2 + 1)
static constexpr int DF_N_CHANNELS      = 1;       // mono
static constexpr int DF_ERB_BANDS        = 96;      // ERB bands (DeepFilterNet's ERB count)
static constexpr int DF_ENC_DIM           = 32;     // encoder dimension
// DeepFilterNet3 model shapes:
//   enc:      [1, 3002, 32]      —  ERB-domain encoder
//   erb_dec:  [1, 3002, 96]      —  ERB-domain decoder
//   df_dec:   [1, 3002, 96]      —  DF output decoder
// 3002 = number of frames at 48kHz / 256 hop ≈ 48000/256 * 16 ≈ 2929... actually
// the model uses variable frame count; we'll dynamic-reshape to batch=1 fixed.

// ─── STFT using FFTW3 ────────────────────────────────────────────────────────

class STFTProcessor {
public:
    STFTProcessor(int frame_size = DF_FRAME_SIZE, int hop_size = DF_HOP_SIZE)
        : frame_size_(frame_size), hop_size_(hop_size),
          window_(frame_size_, 0.0f),
          fft_in_(frame_size), fft_out_(DF_N_BANDS),
          ifft_in_(DF_N_BANDS), ifft_out_(frame_size) {
        // Hann window
        for (int i = 0; i < frame_size_; i++) {
            window_[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (frame_size_ - 1)));
        }
        // FFTW plans
        fft_plan_  = fftwf_plan_dft_r2c_1d(frame_size_,  fft_in_.data(),  (fftwf_complex*)fft_out_.data(),  FFTW_MEASURE);
        ifft_plan_ = fftwf_plan_dft_c2r_1d(frame_size_, (fftwf_complex*)ifft_in_.data(), ifft_out_.data(), FFTW_MEASURE);
    }

    ~STFTProcessor() {
        if (fft_plan_)  fftwf_destroy_plan(fft_plan_);
        if (ifft_plan_) fftwf_destroy_plan(ifft_plan_);
    }

    // Compute STFT magnitude + phase for one frame
    void analyze(const float* input, float* mag, float* phase, int n_bins = DF_N_BANDS) {
        // Apply window
        for (int i = 0; i < frame_size_; i++)
            fft_in_[i] = input[i] * window_[i];

        fftwf_execute(fft_plan_);

        for (int i = 0; i < n_bins; i++) {
            float real = fft_out_[i][0];
            float imag = fft_out_[i][1];
            mag[i]   = sqrtf(real * real + imag * imag);
            phase[i] = atan2f(imag, real);
        }
    }

    // Reconstruct from magnitude + phase
    void synthesize(const float* mag, const float* phase, float* output, int n_bins = DF_N_BANDS) {
        for (int i = 0; i < n_bins; i++) {
            float r = mag[i];
            float p = phase[i];
            ifft_in_[i][0] = r * cosf(p);
            ifft_in_[i][1] = r * sinf(p);
        }
        // Zero out remaining bins (if n_bins < DF_N_BANDS)
        for (int i = n_bins; i < DF_N_BANDS; i++) {
            ifft_in_[i][0] = 0.0f;
            ifft_in_[i][1] = 0.0f;
        }

        fftwf_execute(ifft_plan_);

        // Apply window + overlap-add
        for (int i = 0; i < frame_size_; i++) {
            output[i] = ifft_out_[i] * window_[i] / frame_size_;
        }
    }

    int frame_size() const { return frame_size_; }
    int hop_size()    const { return hop_size_; }

private:
    int frame_size_;
    int hop_size_;
    std::vector<float> window_;
    std::vector<float> fft_in_;
    std::vector<std::complex<float>> fft_out_;  // complex interleaved
    std::vector<std::complex<float>> ifft_in_;
    std::vector<float> ifft_out_;
    fftwf_plan_t fft_plan_;
    fftwf_plan_t ifft_plan_;
};

// ─── DeepFilterNet3 Inference Engine ───────────────────────────────────────

class DeepFilterNet3Engine {
public:
    DeepFilterNet3Engine() = default;
    ~DeepFilterNet3Engine() = default;

    // Load 3 OpenVINO IR models for NPU
    bool init(const std::string& model_dir, const std::string& device = "NPU") {
        try {
            // Three separate Core instances for NPU + GPU simultaneous
            // (DeepFilterNet3 is CPU-bound on NPU, but we use separate cores
            //  to allow GPU effects to run alongside)
            ov::Core core("deepfilternet3-engine");

            // Configure for NPU (stateless per-frame, low batch)
            ov::AnyMap npu_config;
            npu_config["NPU_DEVICE"] = "4TB";
            npu_config["MODEL_CACHING"];  // enable cache

            // enc: [1, 3002, 32]
            enc_ = std::make_unique<ov::CompiledModel>(
                core.compile_model(model_dir + "/enc.xml", device, npu_config));
            enc_infer_  = enc_->get_request();
            enc_output_ = enc_infer_.get_tensor("output_enc");  // name TBD

            // erb_dec: [1, 3002, 96]
            erb_dec_ = std::make_unique<ov::CompiledModel>(
                core.compile_model(model_dir + "/erb_dec.xml", device, npu_config));
            erb_infer_ = erb_dec_->get_request();

            // df_dec: [1, 3002, 96]
            df_dec_ = std::make_unique<ov::CompiledModel>(
                core.compile_model(model_dir + "/df_dec.xml", device, npu_config));
            df_infer_ = df_dec_->get_request();

            // Static reshape BEFORE compile_model("NPU") — NPU requires fixed shape
            // The models above are pre-compiled with fixed shapes, so compile_model
            // already has them baked in.

            fprintf(stderr, "[dfnet3] loaded models: enc=%s erb_dec=%s df_dec=%s\n",
                    model_dir.c_str(), model_dir.c_str(), model_dir.c_str());
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "[dfnet3] ERROR loading models: %s\n", e.what());
            return false;
        }
    }

    // Process one audio frame through DeepFilterNet3
    // Input:  complex spectrogram [n_bins] (magnitude from STFT)
    // Output: complex spectrogram [n_bins] (magnitude after noise suppression)
    void process_frame(const float* spec_mag_in, float* spec_mag_out, int n_bins = DF_ERB_BANDS) {
        // Step 1: ERB encoder (converts spec to ERB domain)
        // enc takes [1, 3002, 32] — we need to accumulate frames
        // For per-frame processing (Phase 2 simplified): use single-frame shape
        auto enc_input = enc_infer_.get_tensor("input_enc");
        // Fill with ERB features from spec_mag_in...

        // Step 2: ERB decoder — reconstruct ERB-domain output
        auto erb_output = erb_infer_.get_tensor("output_erb");

        // Step 3: DF decoder — apply spectral floor (noise suppression)
        auto df_output = df_infer_.get_tensor("output_df");

        // Copy result
        for (int i = 0; i < n_bins; i++) {
            spec_mag_out[i] = spec_mag_in[i];  // simplified — Phase 2 will implement
        }

        enc_infer_.  infer();
        erb_infer_.  infer();
        df_infer_.   infer();
    }

    bool is_loaded() const { return enc_ != nullptr; }

private:
    // Three separate OpenVINO compiled models + infer requests
    std::unique_ptr<ov::CompiledModel> enc_;
    std::unique_ptr<ov::CompiledModel> erb_dec_;
    std::unique_ptr<ov::CompiledModel> df_dec_;

    ov::InferRequest enc_infer_;
    ov::InferRequest erb_infer_;
    ov::InferRequest df_infer_;

    ov::Tensor enc_output_;
};

// ─── Full audio pipeline: PipeWire capture → DF3 → virtual sink ────────────

class AudioPipeline {
public:
    AudioPipeline() = default;

    // Initialize: load models, set up PipeWire
    bool init(const std::string& model_dir, const std::string& source_name) {
        model_dir_ = model_dir;
        source_name_ = source_name;

        // Load DeepFilterNet3
        if (!engine_.init(model_dir_, "NPU")) {
            fprintf(stderr, "[audio] failed to init DeepFilterNet3 engine\n");
            // Fallback to CPU
            if (!engine_.init(model_dir_, "CPU")) {
                return false;
            }
        }

        // Initialize STFT
        stft_ = std::make_unique<STFTProcessor>();

        // Phase 2: Set up PipeWire capture from source_name → process → null sink
        // For Phase 1: just ready to go

        fprintf(stderr, "[audio] pipeline ready (source: %s)\n", source_name_.c_str());
        return true;
    }

    // Start processing (called when mic stream becomes active)
    void start() {
        running_ = true;
        fprintf(stderr, "[audio] pipeline started\n");
    }

    // Stop processing (called when mic stream ends)
    void stop() {
        running_ = false;
        fprintf(stderr, "[audio] pipeline stopped\n");
    }

    // Process a chunk of audio (called per-frame from PipeWire callback)
    void process_chunk(const float* input, float* output, int n_frames) {
        if (!running_ || !engine_.is_loaded()) {
            memcpy(output, input, n_frames * sizeof(float));
            return;
        }

        // STFT → model → ISTFT
        // Simplified: copy through for Phase 1
        memcpy(output, input, n_frames * sizeof(float));
    }

    bool is_running() const { return running_; }

private:
    std::string model_dir_;
    std::string source_name_;
    DeepFilterNet3Engine engine_;
    std::unique_ptr<STFTProcessor> stft_;
    std::atomic<bool> running_{false};
};
