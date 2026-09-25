// studio_effects.cpp — Linux Studio Effects daemon (Phase 1 skeleton).
//
// Architecture: "Auto-activation" mode — differs fundamentally from shlok.asr.
//   - Daemon is event-driven, NOT a polling daemon
//   - Sleeps deeply between events; wakes on:
//     (a) PipeWire stream events (camera/mic start/stop) — via inotify on
//         /run/user/<uid>/pipewire-0/ or via pw_event_loop (Phase 2)
//     (b) State file changes (panel toggles effect ON/OFF) — via inotify
//   - When a camera/mic stream is active AND the corresponding effect is
//     enabled in state files: load model → process → unload when done
//   - No Copilot-key toggle. The panel controls which effects are enabled;
//     the daemon controls WHEN they activate (on stream activity).
//   - Audio feedback (start/stop sounds) fires when streams start/stop.
//   - State files in ~/.local/share/studio-effects/state/
//
// Phase 1: Event-driven daemon skeleton + inotify on state files + signal handling.
// Phase 2: DeepFilterNet3 audio noise suppression pipeline
// Phase 3: MODNet video background blur pipeline
// The daemon sleeps until woken, so it consumes near-zero CPU when idle.
//
#include "audio_pipeline.hpp"
#include "video_pipeline.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/select.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <signal.h>
#include <cmath>

static std::string STATE_DIR  = "/home/shlok/.local/share/studio-effects/state";
static std::string CACHE_DIR  = "/home/shlok/.local/share/studio-effects/cache";
static std::string SOUND_DIR  = "/home/shlok/.local/share/studio-effects/sounds";
static std::string MODELS_DIR = "/home/shlok/.local/share/studio-effects/models";

// ─── Daemon state ──────────────────────────────────────────────────────────
static std::atomic<bool> g_should_exit{false};
static std::atomic<bool> g_reload{false};  // SIGHUP → re-read state files

static std::string g_state_file;
static std::string g_pid_file;
static std::string g_cancel_file;
static std::string g_level_file;

// ─── Pipeline instances (Phase 2+) ───────────────────────────────────────────
static std::unique_ptr<AudioPipeline> g_audio_pipeline;
static std::unique_ptr<VideoPipeline> g_video_pipeline;

// ─── Signal handlers ───────────────────────────────────────────────────────

static void on_sigterm(int) { g_should_exit.store(true); }
static void on_sighup(int) { g_reload.store(true); }

// ─── State file I/O (mirrors dictate.cpp) ──────────────────────────────────

static std::string read_state_str(const std::string& name, const std::string& fallback = "") {
    std::string p = STATE_DIR + "/" + name;
    std::ifstream f(p);
    if (!f) return fallback;
    std::string s;
    std::getline(f, s);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t'))
        s.pop_back();
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.erase(s.begin());
    return s;
}

static bool read_enabled() {
    std::string s = read_state_str("enabled", "on");
    for (char& c : s) c = tolower(c);
    return !(s == "off" || s == "false" || s == "0" || s == "disabled");
}

static std::string read_device() {
    std::string s = read_state_str("device.txt", "NPU");
    if (s.empty()) return "NPU";
    std::string upper = s;
    for (char& c : upper) c = toupper(c);
    if (upper != "CPU" && upper != "GPU" && upper != "NPU") return "NPU";
    return upper;
}

static std::string read_model() {
    return read_state_str("model.txt", "deepfilternet3");
}

static int read_volume() {
    FILE* f = fopen((STATE_DIR + "/volume").c_str(), "r");
    if (!f) return 80;
    int vol = 80;
    if (fscanf(f, "%d", &vol) != 1) vol = 80;
    fclose(f);
    if (vol < 0) vol = 0;
    if (vol > 100) vol = 100;
    return vol;
}

// Per-effect toggle: state/effect_<model> → "on"/"off"
static bool read_effect(const std::string& name) {
    std::string s = read_state_str("effect_" + name, "off");
    for (char& c : s) c = tolower(c);
    return !(s == "off" || s == "false" || s == "0" || s == "disabled");
}

// Piecewise linear gain (same curve as dictate.cpp — user-approved)
static double volume_to_gain(double slider) {
    if (slider <= 0.0)    return 0.0000;
    if (slider >= 100.0)  return 1.0000;
    if (slider >= 50.0)   return 0.6000 + (slider - 50.0)  * (1.0000 - 0.6000) / (100.0 - 50.0);
    if (slider >= 10.0)   return 0.4000 + (slider - 10.0)  * (0.6000 - 0.4000) / (50.0 - 10.0);
    if (slider >= 1.0)    return 0.1000 + (slider - 1.0)   * (0.4000 - 0.1000) / (10.0 - 1.0);
    return 0.1000 * slider / 1.0;
}

// ─── State writers ─────────────────────────────────────────────────────────

static void write_state(const std::string& cls, const std::string& path) {
    if (path.empty()) return;
    std::ofstream f(path);
    f << "{\"alt\":\"" << cls << "\",\"class\":\"" << cls << "\",\"tooltip\":\"\"}\n";
    f.flush();
    system("pkill -RTMIN+8 waybar 2>/dev/null");
}

static void write_state_full(const std::string& cls, const std::string& path,
                             bool stream, time_t since) {
    if (path.empty()) return;
    std::ofstream f(path);
    f << "{\"alt\":\"" << cls << "\",\"class\":\"" << cls << "\",\"tooltip\":\"\"";
    if (stream)  f << ",\"stream\":1";
    if (since > 0) f << ",\"since\":" << since;
    f << "}\n";
    f.flush();
    system("pkill -RTMIN+8 waybar 2>/dev/null");
}

static bool cancel_requested() {
    static bool latched = false;
    if (latched) return true;
    if (::access(g_cancel_file.c_str(), F_OK) == 0) {
        latched = true;
        fprintf(stderr, "[studio] cancel requested via popup X\n"); fflush(stderr);
        return true;
    }
    return false;
}

static void clear_cancel() {
    static bool latched = false;
    latched = false;
    ::unlink(g_cancel_file.c_str());
}

static void write_level_rms(float rms) {
    static auto last_emit = std::chrono::steady_clock::time_point{};
    auto now = std::chrono::steady_clock::now();
    if (now - last_emit < std::chrono::milliseconds(125)) return;
    last_emit = now;
    FILE* f = fopen(g_level_file.c_str(), "w");
    if (f) {
        fprintf(f, "%.6f\n", rms);
        fclose(f);
    }
}

// ─── Audio feedback (double-fork + setsid + paplay, mirrors dictate.cpp) ───

static void play_sound_async(const char* label, const char* which) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.wav", SOUND_DIR.c_str(), which);
    if (access(path, R_OK) != 0) {
        fprintf(stderr, "[studio-audio] %s: sound file not found, skipping\n", label);
        fflush(stderr);
        return;
    }
    int slider = read_volume();
    double gain = volume_to_gain((double)slider);

    char pw_vol[32];
    snprintf(pw_vol, sizeof(pw_vol), "--volume=%.4f", gain);

    // Block SIGCHLD around fork
    sigset_t set, oldset;
    sigemptyset(&set);
    sigaddset(&set, SIGCHLD);
    sigaddset(&set, SIGHUP);
    sigprocmask(SIG_BLOCK, &set, &oldset);

    pid_t mid = fork();
    if (mid < 0) { sigprocmask(SIG_SETMASK, &oldset, NULL); return; }
    if (mid > 0) {
        int status; waitpid(mid, &status, WNOHANG);
        sigprocmask(SIG_SETMASK, &oldset, NULL);
        return;
    }
    // Middle child
    sigprocmask(SIG_SETMASK, &oldset, NULL);
    setsid();

    pid_t player = fork();
    if (player < 0) _exit(0);
    if (player > 0) _exit(0);

    // Grandchild: play sound
    execlp("pw-play", "pw-play", pw_vol, path, (char*)nullptr);
    execlp("paplay", "paplay", path, (char*)nullptr);
    _exit(127);
}

static void play_start_sound() { play_sound_async("start", "start"); }
static void play_stop_sound()  { play_sound_async("stop", "stop"); }

// ─── Stream detection (Phase 2: real PipeWire monitoring) ──────────────────
// Phase 2: monitor PipeWire for camera/mic activity via the PipeWire event
// loop or by polling /proc or /sys. For Phase 1, we stub this to return false.
// The daemon still runs and watches state files via inotify.

static bool camera_in_use() {
    // Phase 2: check PipeWire for active camera source nodes
    return false;
}

static bool mic_in_use() {
    // Phase 2: check PipeWire for active audio input
    return false;
}

// ─── Main daemon loop (event-driven, low-power) ─────────────────────────────
// Sleeps in select() until woken by:
//   - inotify: state file changed (panel toggled an effect)
//   - inotify: PipeWire state changed (Phase 2: stream active/inactive)
//   - SIGTERM/SIGINT: shutdown
//   - SIGHUP: reload (re-read config)
// Does NOT poll every 100ms — stays blocked until an event arrives.

static void run_daemon() {
    g_state_file   = STATE_DIR + "/status.json";
    g_pid_file     = STATE_DIR + "/studio-effects.pid";
    g_cancel_file  = STATE_DIR + "/cancel-requested";
    g_level_file   = STATE_DIR + "/level";

    mkdir(STATE_DIR.c_str(), 0755);

    // Write PID file
    {
        FILE* f = fopen(g_pid_file.c_str(), "w");
        if (f) { fprintf(f, "%d\n", getpid()); fclose(f); }
    }

    // Signal handlers
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigterm;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);

    struct sigaction sh;
    memset(&sh, 0, sizeof(sh));
    sh.sa_handler = on_sighup;
    sigaction(SIGHUP, &sh, nullptr);

    // Set up inotify for state file changes
    int inotify_fd = inotify_init1(IN_NONBLOCK);
    if (inotify_fd < 0) {
        fprintf(stderr, "[studio] WARNING: inotify init failed, using fallback\n"); fflush(stderr);
    }

    // Watch the state directory for changes
    int state_wd = -1;
    if (inotify_fd >= 0) {
        state_wd = inotify_add_watch(inotify_fd, STATE_DIR.c_str(),
                                     IN_MODIFY | IN_CREATE | IN_DELETE);
        if (state_wd < 0) {
            fprintf(stderr, "[studio] WARNING: inotify watch on state dir failed\n"); fflush(stderr);
        } else {
            fprintf(stderr, "[studio] watching state dir for changes\n"); fflush(stderr);
        }
    }

    // Also watch for PipeWire events via /run/user
    // Phase 2: will add inotify on /run/user/<uid>/pipewire-0/ for stream events
    int pipewire_fd = -1;
    int pipewire_wd = -1;
    char pipewire_path[256];
    snprintf(pipewire_path, sizeof(pipewire_path), "/run/user/%d", getuid());
    pipewire_fd = inotify_init1(IN_NONBLOCK);
    if (pipewire_fd >= 0) {
        pipewire_wd = inotify_add_watch(pipewire_fd, "/run/user", IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO);
        // Will add deeper watches in Phase 2
    }

    write_state("idle", g_state_file);
    fprintf(stderr, "[studio] daemon ready (PID %d)\n", getpid());
    fprintf(stderr, "[studio] Auto-activation: effects load when mic/camera active + panel enabled\n");
    fprintf(stderr, "[studio] Event-driven: sleeping until state change or stream event\n");
    fflush(stderr);

    bool was_audio_active = false;
    bool was_video_active = false;

    // Main event loop: sleep until woken by inotify or signal
    for (;;) {
        if (g_should_exit.load()) break;
        if (g_reload.exchange(false)) {
            fprintf(stderr, "[studio] reload signal received\n"); fflush(stderr);
            // Re-read offload setting, reload models if needed
        }

        // Build fd_set for select
        fd_set rfds;
        FD_ZERO(&rfds);
        int max_fd = 0;

        if (inotify_fd >= 0) {
            FD_SET(inotify_fd, &rfds);
            max_fd = std::max(max_fd, inotify_fd);
        }
        if (pipewire_fd >= 0) {
            FD_SET(pipewire_fd, &rfds);
            max_fd = std::max(max_fd, pipewire_fd);
        }

        // Sleep until event or timeout (30s timeout as safety net)
        struct timeval tv;
        tv.tv_sec = 30;
        tv.tv_usec = 0;

        int ret = select(max_fd + 1, &rfds, nullptr, nullptr, &tv);

        if (g_should_exit.load()) break;

        if (ret <= 0) {
            // Timeout or error — re-check state
            continue;
        }

        // Drain inotify events (we just need to know SOMETHING changed)
        char buf[4096];
        if (inotify_fd >= 0 && FD_ISSET(inotify_fd, &rfds)) {
            read(inotify_fd, buf, sizeof(buf));  // drain
        }
        if (pipewire_fd >= 0 && FD_ISSET(pipewire_fd, &rfds)) {
            read(pipewire_fd, buf, sizeof(buf));  // drain
            // Phase 2: check if pipewire-0/ directory appeared
            // Re-create watch if needed
        }

        // Re-read state from files
        bool enabled = read_enabled();
        if (!enabled) {
            // System disabled — unload all pipelines
            if (was_audio_active || was_video_active) {
                if (g_audio_pipeline) g_audio_pipeline->stop();
                if (g_video_pipeline) {}  // video stop is implicit
                g_audio_pipeline = nullptr;
                g_video_pipeline = nullptr;
                was_audio_active = false;
                was_video_active = false;
                write_state("idle", g_state_file);
            }
            continue;
        }

        bool ns_on   = read_effect("deepfilternet3");
        bool blur_on = read_effect("modnet");
        bool af_on   = read_effect("face-adas");
        bool vsr_on  = read_effect("audiosr-speech");
        bool isr_on  = read_effect("realesrgan-x4");

        // --- Audio pipeline (DeepFilterNet3) ---
        bool mic_active = mic_in_use();
        if (ns_on && mic_active) {
            if (!g_audio_pipeline) {
                // Initialize audio pipeline
                std::string audio_src = read_state_str("audio_source", "@DEFAULT_AUDIO_SOURCE@");
                std::string model_dir = MODELS_DIR + "/deepfilternet3";
                g_audio_pipeline = std::make_unique<AudioPipeline>();
                if (g_audio_pipeline->init(model_dir, audio_src)) {
                    g_audio_pipeline->start();
                    if (!was_audio_active) {
                        play_start_sound();
                        fprintf(stderr, "[studio] audio: DeepFilterNet3 active on mic\n");
                        fflush(stderr);
                    }
                } else {
                    g_audio_pipeline = nullptr;
                    fprintf(stderr, "[studio] audio: pipeline init failed\n");
                    fflush(stderr);
                }
            }
        } else {
            // Mic not in use, or noise suppression disabled
            if (g_audio_pipeline) {
                g_audio_pipeline->stop();
                g_audio_pipeline = nullptr;
                if (was_audio_active) {
                    play_stop_sound();
                    fprintf(stderr, "[studio] audio: DeepFilterNet3 stopped\n");
                    fflush(stderr);
                }
            }
        }
        was_audio_active = (g_audio_pipeline && g_audio_pipeline->is_running());

        // --- Video pipeline (MODNet background blur) ---
        bool cam_active = camera_in_use();
        if (blur_on && cam_active) {
            if (!g_video_pipeline) {
                // Initialize video pipeline
                VideoPipeline::Config vcfg;
                vcfg.model_dir    = MODELS_DIR + "/modnet";
                vcfg.blur_strength = std::stof(read_state_str("blur_strength", "50"));
                vcfg.bg_image_path  = read_state_str("bg_image_path", "");

                g_video_pipeline = std::make_unique<VideoPipeline>();
                if (g_video_pipeline->init(vcfg)) {
                    if (!was_video_active) {
                        play_start_sound();
                        fprintf(stderr, "[studio] video: MODNet blur active on camera\n");
                        fflush(stderr);
                    }
                } else {
                    g_video_pipeline = nullptr;
                    fprintf(stderr, "[studio] video: pipeline init failed\n");
                    fflush(stderr);
                }
            }
        } else {
            // Camera not in use, or blur disabled
            if (g_video_pipeline) {
                g_video_pipeline = nullptr;
                if (was_video_active) {
                    play_stop_sound();
                    fprintf(stderr, "[studio] video: MODNet blur stopped\n");
                    fflush(stderr);
                }
            }
        }
        was_video_active = (g_video_pipeline != nullptr);

        // Update status
        if (was_audio_active || was_video_active) {
            write_state_full("recording", g_state_file, false, time(nullptr));
        } else if (enabled) {
            write_state("idle", g_state_file);
        }
    }

    // Cleanup
    if (inotify_fd >= 0) close(inotify_fd);
    if (pipewire_fd >= 0) close(pipewire_fd);
    g_audio_pipeline = nullptr;
    g_video_pipeline = nullptr;
    ::unlink(g_pid_file.c_str());
    write_state("idle", g_state_file);
    fprintf(stderr, "[studio] daemon shutting down\n");
    fflush(stderr);
}

// ─── Command handlers ────────────────────────────────────────────────────────

static void handle_record_test() {
    fprintf(stderr, "[studio] --record-test: not yet implemented (Phase 2)\n"); fflush(stderr);
}

static void handle_play_test() {
    fprintf(stderr, "[studio] --play-test: not yet implemented (Phase 2)\n"); fflush(stderr);
}

static void handle_test_camera() {
    fprintf(stderr, "[studio] --test-camera: serving test-page\n"); fflush(stderr);
    // Serve the camera.html test page on port 18080
    // Phase 1: just verify the file exists
    std::string path = "/home/shlok/.local/share/studio-effects/tools/test-page/camera.html";
    if (access(path.c_str(), R_OK) == 0) {
        fprintf(stderr, "[studio] test page available at: file://%s/camera.html\n", path.c_str()); fflush(stderr);
        fprintf(stderr, "[studio] open http://localhost:18080/camera.html once the HTTP server is running\n"); fflush(stderr);
    } else {
        fprintf(stderr, "[studio] no test page found\n"); fflush(stderr);
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string mode;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--daemon") mode = "daemon";
        else if (a == "--record-test") mode = "record_test";
        else if (a == "--play-test") mode = "play_test";
        else if (a == "--test-camera") mode = "test_camera";
        else if (a == "--state-dir" && i + 1 < argc) STATE_DIR = argv[++i];
        else if (a == "--cache-dir" && i + 1 < argc) CACHE_DIR = argv[++i];
        else if (a == "--help" || a == "-h" || a == "help") {
            fprintf(stderr,
                "studio-effects — Linux Studio Effects (auto-activation daemon)\n\n"
                "Usage:\n"
                "  studio-effects --daemon         Run as background daemon (event-driven)\n"
                "  studio-effects --record-test    Test audio capture (Phase 2)\n"
                "  studio-effects --play-test      Test audio output (Phase 2)\n"
                "  studio-effects --test-camera    Show camera test page info\n\n"
                "Effects auto-activate: when mic or camera stream is active\n"
                "AND the corresponding effect is enabled via the panel.\n"
                "No manual toggle needed — the daemon is always-on.\n");
            return 0;
        }
    }

    if (mode.empty()) {
        fprintf(stderr,
            "Usage: studio-effects --daemon | --record-test | --play-test | --test-camera\n"
            "Run with --help for details.\n");
        return 1;
    }

    if (mode == "daemon") {
        run_daemon();
        return 0;
    }
    if (mode == "record_test") {
        handle_record_test();
        return 0;
    }
    if (mode == "play_test") {
        handle_play_test();
        return 0;
    }
    if (mode == "test_camera") {
        handle_test_camera();
        return 0;
    }

    // Should not reach here
    return 1;
}
