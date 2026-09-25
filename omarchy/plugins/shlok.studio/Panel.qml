// shlok.studio — Linux Studio Effects control panel for the Omarchy bar.
//
// Mirrors the shlok.asr panel architecture but controls five audio/video
// effects instead of ASR models. Each effect has ON/OFF + its specific
// controls. The daemon reads state files on every toggle.
//
// State files (under ~/.local/share/studio-effects/state/):
//   enabled        → "on"/"off" (master toggle — controls ALL effects)
//   device.txt     → "CPU"/"GPU"/"NPU"
//   effect_<name>  → "on"/"off" per-effect toggle (deepfilternet3, modnet, face-adas, audiosr-speech, realesrgan-x4)
//   blur_strength  → 0-100 (modnet only)
//   bg_image_path  → file path (modnet only)
//   audiosr_preset → "low"/"med"/"high"
//   realesrgan_q   → 0-100 (single quality slider)
//   audio_source   → PipeWire source node name
//   status.json    → {"class":"idle|recording|processing","since":epoch}
//   volume         → 0-100 (audio feedback)
//   offload        → "1 30" / "0 <secs>"
//   level          → plain RMS (for VU bar in popup)
//   daemon.pid     → PID of running daemon
//   cancel-requested → empty file (popup X button)
import QtQuick
import QtQuick.Effects
import QtQuick.Layouts
import Quickshell
import Quickshell.Io
import qs.Ui
import qs.Commons
import "Model.js" as Model

Panel {
  id: root
  moduleName: "shlok.studio"
  ipcTarget: "shlok.studio"

  readonly property string stateDir: "/home/shlok/.local/share/studio-effects/state"
  readonly property string modelsDir: "/home/shlok/.local/share/studio-effects/models"

  // --- Live state mirrored from files ---
  // System enabled/disabled: daemon is started/stopped via systemctl or wrapper script.
  // This panel toggles per-effect ON/OFF — effects auto-activate when streams are active.
  property bool systemEnabled: true
  property string activeDevice: "NPU"
  property string statusClass: "idle"

  // Per-effect toggles
  property bool noiseSuppressionEnabled: true
  property bool backgroundBlurEnabled: false
  property bool autoFramingEnabled: false
  property bool voiceSrEnabled: false
  property bool imageSrEnabled: false

  // Per-effect controls
  property int blurStrength: 50        // 0-100 (modnet)
  property string bgImagePath: ""      // modnet bg image
  property string audiosrPreset: "med" // low / med / high
  property int realesrganQuality: 50   // 0-100 (single slider)

  // Audio input device
  property string audioSource: "@DEFAULT_AUDIO_SOURCE@"
  property var audioSources: []

  // --- Process pool for writing settings ---
  Process { id: actionProc; onExited: {} }

  function writeSetting(shellCmd) {
    if (actionProc.running) return
    actionProc.command = ["bash", "-c", shellCmd]
    actionProc.running = true
  }

  function setSystemEnabled(on) {
    systemEnabled = on
    writeSetting("mkdir -p " + stateDir + " && echo " + (on ? "on" : "off") + " > " + stateDir + "/enabled")
  }
  function setDevice(d) {
    activeDevice = d
    writeSetting("echo " + d + " > " + stateDir + "/device.txt")
  }
  function setEffect(name, on) {
    writeSetting("echo " + (on ? "on" : "off") + " > " + stateDir + "/effect_" + name)
  }
  function setNoiseSuppression(on) {
    noiseSuppressionEnabled = on
    setEffect("deepfilternet3", on)
  }
  function setBackgroundBlur(on) {
    backgroundBlurEnabled = on
    setEffect("modnet", on)
  }
  function setAutoFraming(on) {
    autoFramingEnabled = on
    setEffect("face-adas", on)
  }
  function setVoiceSr(on) {
    voiceSrEnabled = on
    setEffect("audiosr-speech", on)
  }
  function setImageSr(on) {
    imageSrEnabled = on
    setEffect("realesrgan-x4", on)
  }
  function setBlurStrength(v) {
    blurStrength = v
    writeSetting("echo " + v + " > " + stateDir + "/blur_strength")
  }
  function setBgImage(path) {
    bgImagePath = path
    writeSetting("echo '" + path + "' > " + stateDir + "/bg_image_path")
  }
  function setAudiosrPreset(p) {
    audiosrPreset = p
    writeSetting("echo " + p + " > " + stateDir + "/audiosr_preset")
  }
  function setRealEsrganQuality(v) {
    realesrganQuality = v
    writeSetting("echo " + v + " > " + stateDir + "/realesrgan_q")
  }
  function setAudioSource(src) {
    audioSource = src
    writeSetting("echo '" + src + "' > " + stateDir + "/audio_source")
  }
  function setVolume(v) {
    writeSetting("echo " + v + " > " + stateDir + "/volume")
  }
  function setOffload(idx) {
    writeSetting("echo '" + (idx === 0 ? "1 30" : "0 " + offloadSecs[idx]) + "' > " + stateDir + "/offload")
  }

  // --- Offload slider ---
  readonly property var offloadStops: ["Immediate", "30s", "1m", "2m", "5m", "10m", "15m", "Never"]
  readonly property var offloadSecs:  [30, 30, 60, 120, 300, 600, 900, 31536000]

  // --- Glyphs ---
  readonly property string idleGlyph: "󰋧"
  readonly property string recordingGlyph: "󰏚"
  readonly property string transcribeGlyph: ""
  readonly property string errorGlyph: "󰀨"

  readonly property string barGlyph:
    root.statusClass === "recording"     ? root.recordingGlyph :
    root.statusClass === "processing"    ? root.transcribeGlyph :
                                          root.idleGlyph

  readonly property bool barDimmed: !root.systemEnabled

  // --- Read state files ---
  function refresh() {
    if (!procEnabled.running) procEnabled.running = true
    if (!procDevice.running) procDevice.running = true
    if (!procStatus.running) procStatus.running = true
    if (!procEffects.running) procEffects.running = true
    if (!procBlur.running) procBlur.running = true
    if (!procPreset.running) procPreset.running = true
    if (!procSrQ.running) procSrQ.running = true
    if (!procAudioSrc.running) procAudioSrc.running = true
    if (!procVolume.running) procVolume.running = true
    if (!procOffload.running) procOffload.running = true
    if (!procSources.running) procSources.running = true
  }

  Process { id: procEnabled
    command: ["cat", stateDir + "/enabled"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.systemEnabled = Model.boolFromFile(text) }
  }
  Process { id: procDevice
    command: ["cat", stateDir + "/device.txt"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.activeDevice = Model.deviceFromFile(text) }
  }
  Process { id: procStatus
    command: ["cat", stateDir + "/status.json"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var m = /"class"\s*:\s*"([^"]*)"/.exec(String(text || ""))
      root.statusClass = m ? m[1] : "idle"
    } }
  }
  Process { id: procEffects
    command: ["bash", "-c", "for e in deepfilternet3 modnet face-adas audiosr-speech realesrgan-x4; do cat " + stateDir + "/effect_$e 2>/dev/null; echo -n '|'; done"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var parts = String(text || "").trim().split("|")
      if (parts.length >= 5) {
        root.noiseSuppressionEnabled = parts[0] !== "off"
        root.backgroundBlurEnabled = parts[1] !== "off"
        root.autoFramingEnabled = parts[2] !== "off"
        root.voiceSrEnabled = parts[3] !== "off"
        root.imageSrEnabled = parts[4] !== "off"
      }
    } }
  }
  Process { id: procBlur
    command: ["bash", "-c", "cat " + stateDir + "/blur_strength 2>/dev/null || echo 50"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var v = parseInt(String(text || "50").trim())
      root.blurStrength = isNaN(v) ? 50 : Math.max(0, Math.min(100, v))
    } }
  }
  Process { id: procPreset
    command: ["bash", "-c", "cat " + stateDir + "/audiosr_preset 2>/dev/null || echo med"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.audiosrPreset = String(text || "").trim() }
  }
  Process { id: procSrQ
    command: ["bash", "-c", "cat " + stateDir + "/realesrgan_q 2>/dev/null || echo 50"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var v = parseInt(String(text || "50").trim())
      root.realesrganQuality = isNaN(v) ? 50 : Math.max(0, Math.min(100, v))
    } }
  }
  Process { id: procAudioSrc
    command: ["bash", "-c", "cat " + stateDir + "/audio_source 2>/dev/null || echo @DEFAULT_AUDIO_SOURCE@"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.audioSource = String(text || "").trim() }
  }
  Process { id: procVolume
    command: ["bash", "-c", "cat " + stateDir + "/volume 2>/dev/null || echo 80"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var v = parseInt(String(text || "80").trim())
      root.feedbackVolume = isNaN(v) ? 80 : Math.max(0, Math.min(100, v))
    } }
  }
  property int feedbackVolume: 80
  Process { id: procOffload
    command: ["bash", "-c", "cat " + stateDir + "/offload 2>/dev/null || echo none"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var t = String(text || "").trim().split(/\s+/)
      if (t.length >= 1 && t[0] !== "none") {
        var secs = (t.length >= 2 ? parseInt(t[1]) : 30) || 30
        root.offloadIndex = root.offloadIndexFromPolicy(t[0] === "1", secs)
      }
    } }
  }
  property int offloadIndex: 0
  function offloadIndexFromPolicy(immediate, secs) {
    if (immediate) return 0
    for (var i = 1; i < root.offloadSecs.length; i++)
      if (root.offloadSecs[i] === secs) return i
    var nearest = 1, best = Math.abs(root.offloadSecs[1] - secs)
    for (var j = 2; j < root.offloadSecs.length; j++) {
      var d = Math.abs(root.offloadSecs[j] - secs)
      if (d < best) { best = d; nearest = j }
    }
    return nearest
  }
  Process { id: procSources
    command: ["bash", "-c", "pactl list sources short 2>/dev/null | awk '{print $2}' | head -20"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      root.audioSources = String(text || "").trim().split("\n").filter(function(s) { return s.length > 0 })
    } }
  }

  // --- Live status poll (always on) ---
  Timer {
    id: liveStatusTimer
    interval: 800
    running: true
    repeat: true
    onTriggered: {
      if (!procStatus.running) procStatus.running = true
    }
  }
  Timer { interval: 2500; running: root.opened; repeat: true; onTriggered: root.refresh() }

  // --- Bar icon ---
  readonly property bool barActive: root.statusClass === "recording" || root.statusClass === "processing"
  visible: true
  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  BarIconButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: root.barGlyph
    tooltipText: root.systemEnabled
      ? (root.statusClass === "idle" ? "Studio Effects ready · " + root.activeDevice : "Studio Effects " + root.statusClass)
      : "Studio Effects off"
    useActiveColor: false
    active: root.barActive
    dimmed: root.barDimmed
    onPressed: function() { root.toggle() }
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    contentWidth: panel.fittedContentWidth(Style.space(380))
    contentHeight: panel.fittedContentHeight(column.implicitHeight)

    PanelKeyCatcher { id: keyCatcher; anchors.fill: parent }

    Column {
      id: column
      anchors.left: parent.left
      anchors.right: parent.right
      anchors.top: parent.top
      spacing: Style.space(14)

      // ---------- Hero row ----------
      // System ON/OFF is managed by daemon start/stop (not a panel toggle).
      // Clicking the bar icon opens this panel; individual effects are toggled below.
      Item {
        width: parent.width
        implicitHeight: Math.max(heroIcon.implicitHeight, heroLabels.implicitHeight, statusBadge.implicitHeight)

        Text {
          id: heroIcon
          textFormat: Text.PlainText
          text: root.barGlyph
          color: root.bar.foreground
          font.family: root.bar.fontFamily
          font.pixelSize: Style.font.display
          opacity: root.barDimmed ? 0.5 : 1.0
          anchors.left: parent.left
          anchors.verticalCenter: parent.verticalCenter
        }

        // Status badge (ON / OFF indicator, not a toggle)
        Item {
          id: statusBadge
          anchors.right: parent.right
          anchors.verticalCenter: parent.verticalCenter
          implicitWidth: badgeText.implicitWidth + Style.space(8) * 2
          implicitHeight: badgeText.implicitHeight + Style.space(4) * 2
          radius: Style.cornerRadius
          color: root.systemEnabled
            ? Style.selectedFillFor(root.bar.foreground, root.bar.accent).backgroundColor
            : "transparent"
          border.color: root.systemEnabled ? root.bar.accent : root.bar.foreground
          border.width: root.systemEnabled ? 1 : 1
          clip: true

          Text {
            id: badgeText
            anchors.centerIn: parent
            text: root.systemEnabled ? "ON" : "OFF"
            color: root.systemEnabled ? root.bar.foreground : Qt.darker(root.bar.foreground, 1.6)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.caption
            font.bold: true
          }
        }

        Column {
          id: heroLabels
          anchors.left: heroIcon.right
          anchors.leftMargin: Style.space(14)
          anchors.right: parent.right
          anchors.rightMargin: statusBadge.width + Style.space(12)
          anchors.verticalCenter: parent.verticalCenter
          spacing: Style.space(2)

          Text {
            text: "STUDIO EFFECTS"
            color: root.bar.foreground
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.title
            font.bold: true
            elide: Text.ElideRight
            width: parent.width
          }
          Text {
            id: heroStatus
            textFormat: Text.PlainText
            text: root.systemEnabled
              ? root.statusClass.toUpperCase()
              : "OFF"
            color: Qt.darker(root.bar.foreground, 1.4)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.caption
            font.bold: true
            elide: Text.ElideRight
            width: parent.width
          }
        }
      }

      // ---------- Accelerator row ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        Item {
          width: parent.width
          readonly property real intelLogoH: Style.font.bodySmall * 1.6
          implicitHeight: Math.max(accHeader.implicitHeight, intelLogoH)

          PanelSectionHeader {
            id: accHeader
            text: "ACCELERATOR"
            foreground: root.bar.foreground
            fontFamily: root.bar.fontFamily
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
          }

          Item {
            id: poweredByRow
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            readonly property real logoW: Math.round(accHeader.intelLogoH * 24.24 / 9.552)
            width: poweredByText.implicitWidth + Style.space(8) + logoW
            height: Math.max(poweredByText.implicitHeight, accHeader.intelLogoH + 1)

            Text {
              id: poweredByText
              text: "POWERED BY"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.bodySmall
              anchors.left: parent.left
              anchors.bottom: parent.bottom
            }
            Item {
              width: poweredByRow.logow
              height: accHeader.intelLogoH
              anchors.right: parent.right
              anchors.bottom: parent.bottom
              anchors.bottomMargin: Math.round(accHeader.intelLogoH * 2 / 31) + 1
              Image {
                id: intelSvg
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                sourceSize.height: accHeader.intelLogoH * 2
                source: Qt.resolvedUrl("assets/intel.svg")
              }
              MultiEffect {
                anchors.fill: intelSvg
                source: intelSvg
                colorization: 1.0
                colorizationColor: root.bar.foreground
              }
            }
          }
        }

        Row {
          id: deviceRow
          width: parent.width
          spacing: Style.space(6)
          readonly property int count: 3
          readonly property real cellWidth: (width - spacing * (count - 1)) / count

          Repeater {
            model: ["CPU", "GPU", "NPU"]
            delegate: BorderSurface {
              property string dev: modelData
              width: deviceRow.cellWidth
              height: Style.font.title * 2 + Style.space(2) * 2
              color: root.activeDevice === modelData
                ? Style.selectedFillFor(root.bar.foreground, root.bar.accent)
                : "transparent"
              borderSpec: Border.controlSpec("hover-cursor", root.bar.foreground, root.bar.accent)
              border.color: Border.canUseNative(borderSpec) ? Border.color(borderSpec) : "transparent"
              border.width: Border.canUseNative(borderSpec) ? Border.uniformWidth(borderSpec) : 0
              opacity: 1.0
              radius: Style.cornerRadius

              Row {
                anchors.centerIn: parent
                spacing: Style.space(8)
                Item {
                  width: (modelData === "GPU") ? Style.font.title * 2.5 : Style.font.title * 2
                  height: Style.font.title * 2
                  anchors.verticalCenter: parent.verticalCenter
                  clip: true
                  Image {
                    anchors.fill: parent
                    source: Qt.resolvedUrl("assets/" + modelData.toLowerCase() + ".svg")
                    fillMode: Image.PreserveAspectFit
                    sourceSize.height: Style.font.title * 4
                  }
                  MultiEffect {
                    anchors.fill: parent
                    source: parent
                    colorization: 1.0
                    colorizationColor: root.bar.foreground
                  }
                }
                Text {
                  text: modelData
                  color: root.bar.foreground
                  font.family: root.bar.fontFamily
                  font.pixelSize: Style.font.title
                  font.bold: true
                  anchors.verticalCenter: parent.verticalCenter
                  anchors.verticalCenterOffset: 1
                }
              }

              MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                hoverEnabled: true
                onClicked: root.setDevice(modelData)
              }
            }
          }
        }
      }

      // ---------- EFFECT TOGGLES ----------
      function effectToggle(name, enabled, setFn) {
        // Helper to draw an effect row
      }

      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        PanelSectionHeader { text: "EFFECTS"; foreground: root.bar.foreground; fontFamily: root.bar.fontFamily }

        // --- Noise Suppression ---
        Item {
          width: parent.width
          implicitHeight: Style.font.title * 2
          Text {
            id: nsLabel
            anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
            text: "Noise Suppression"
            color: root.bar.foreground
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.body
          }
          ToggleSwitch {
            id: nsToggle
            checked: root.noiseSuppressionEnabled
            foreground: root.bar.foreground
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            onToggled: root.setNoiseSuppression(!root.noiseSuppressionEnabled)
          }
        }

        // --- Background Blur ---
        Column {
          width: parent.width
          spacing: Style.space(8)

          Item {
            width: parent.width
            implicitHeight: Style.font.title * 2
            Text {
              anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
              text: "Background Blur"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.body
            }
            ToggleSwitch {
              id: bbToggle
              checked: root.backgroundBlurEnabled
              foreground: root.bar.foreground
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              onToggled: root.setBackgroundBlur(!root.backgroundBlurEnabled)
            }
          }

          // Blur strength slider (only visible when blur is on)
          Item {
            width: parent.width
            implicitHeight: Style.font.title * 2
            visible: root.backgroundBlurEnabled
            Text {
              id: blurLabel
              anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
              text: "Blur strength  " + root.blurStrength + "%"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.caption
            }
            PanelSlider {
              id: blurSlider
              bar: root.bar
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              width: Style.space(120)
              minimum: 0
              maximum: 100
              step: 1
              integer: true
              value: root.blurStrength
              onReleased: function(v) { root.setBlurStrength(Math.round(v)) }
              opacity: root.backgroundBlurEnabled ? 1.0 : 0.0
            }
          }

          // Background image picker
          Item {
            width: parent.width
            implicitHeight: Style.font.title * 2
            visible: root.backgroundBlurEnabled
            Text {
              anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
              text: "Background image"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.body
            }
            MouseArea {
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              cursorShape: Qt.PointingHandCursor
              hoverEnabled: true
              Text {
                anchors.centerIn: parent
                text: root.bgImagePath ? "✎ Change" : "Choose…"
                color: root.bar.foreground
                font.family: root.bar.fontFamily
                font.pixelSize: Style.font.caption
                font.bold: true
              }
              onClicked: {
                // Write default placeholder — user can edit the file directly
                var path = root.stateDir + "/bg_image_path"
                root.writeSetting("echo '' > " + path + " && echo 'Edit " + path + " to set bg image path'")
              }
            }
          }
        }

        // --- Auto-Framing ---
        Item {
          width: parent.width
          implicitHeight: Style.font.title * 2
          Text {
            anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
            text: "Auto-Framing"
            color: root.bar.foreground
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.body
          }
          ToggleSwitch {
            id: afToggle
            checked: root.autoFramingEnabled
            foreground: root.bar.foreground
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            onToggled: root.setAutoFraming(!root.autoFramingEnabled)
          }
        }

        // --- Voice SR (AudioSR) ---
        Column {
          width: parent.width
          spacing: Style.space(8)

          Item {
            width: parent.width
            implicitHeight: Style.font.title * 2
            Text {
              anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
              text: "Voice Super-Resolution"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.body
            }
            ToggleSwitch {
              id: vsrToggle
              checked: root.voiceSrEnabled
              foreground: root.bar.foreground
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              onToggled: root.setVoiceSr(!root.voiceSrEnabled)
            }
          }

          // Quality preset dropdown (only visible when VSR is on)
          Item {
            width: parent.width
            implicitHeight: Style.font.title * 2
            visible: root.voiceSrEnabled
            Text {
              anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
              text: "Quality"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.caption
            }
            // Simple text-based preset selector
            Text {
              id: presetLabel
              anchors.left: parent.left
              anchors.bottom: parent.bottom
              text: "Preset: " + root.audiosrPreset.toUpperCase() + " (tap to cycle)"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.caption
              font.bold: true
            }
            MouseArea {
              anchors.fill: parent
              onClicked: {
                var presets = ["low", "med", "high"]
                var idx = presets.indexOf(root.audiosrPreset)
                var next = presets[(idx + 1) % presets.length]
                root.setAudiosrPreset(next)
              }
            }
          }
        }

        // --- Image SR (Real-ESRGAN) ---
        Column {
          width: parent.width
          spacing: Style.space(8)

          Item {
            width: parent.width
            implicitHeight: Style.font.title * 2
            Text {
              anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
              text: "Image Upscaling"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.body
            }
            ToggleSwitch {
              id: isrToggle
              checked: root.imageSrEnabled
              foreground: root.bar.foreground
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              onToggled: root.setImageSr(!root.imageSrEnabled)
            }
          }

          // Quality slider (only visible when image SR is on)
          Item {
            width: parent.width
            implicitHeight: Style.font.title * 2
            visible: root.imageSrEnabled
            Text {
              id: qLabel
              anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
              text: "Quality  " + root.realesrganQuality + "%"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.caption
            }
            PanelSlider {
              id: qSlider
              bar: root.bar
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              width: Style.space(120)
              minimum: 0
              maximum: 100
              step: 1
              integer: true
              value: root.realesrganQuality
              onReleased: function(v) { root.setRealEsrganQuality(Math.round(v)) }
              opacity: root.imageSrEnabled ? 1.0 : 0.0
            }
          }
        }
      }

      // ---------- Audio input device ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        PanelSectionHeader { text: "MICROPHONE"; foreground: root.bar.foreground; fontFamily: root.bar.fontFamily }

        Text {
          text: root.audioSource || "@DEFAULT_AUDIO_SOURCE@"
          color: root.bar.foreground
          font.family: root.bar.fontFamily
          font.pixelSize: Style.font.caption
          elide: Text.ElideRight
          width: parent.width
        }

        Text {
          text: root.audioSources.length > 0 ? "Tap to cycle input (" + root.audioSources.length + " found)" : "Scanning sources..."
          color: Qt.darker(root.bar.foreground, 1.4)
          font.family: root.bar.fontFamily
          font.pixelSize: Style.font.caption
        }

        MouseArea {
          anchors.fill: parent
          onClicked: {
            if (root.audioSources.length > 0) {
              var idx = root.audioSources.indexOf(root.audioSource)
              idx = (idx + 1) % root.audioSources.length
              root.setAudioSource(root.audioSources[idx])
            }
            if (!procSources.running) procSources.running = true
          }
        }
      }

      // ---------- Model offload ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        Item {
          width: parent.width
          implicitHeight: Style.font.body * 1.4

          PanelSectionHeader {
            text: "MODEL OFFLOAD"
            foreground: root.bar.foreground
            fontFamily: root.bar.fontFamily
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
          }
          Text {
            text: root.offloadStops[root.offloadIndex]
            color: Qt.darker(root.bar.foreground, 1.4)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.caption
            font.bold: true
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
          }
        }

        PanelSlider {
          anchors.fill: parent
          bar: root.bar
          minimum: 0
          maximum: root.offloadStops.length - 1
          step: 1
          integer: true
          tickCount: root.offloadStops.length
          value: root.offloadIndex
          onReleased: function(v) {
            root.offloadIndex = Math.round(v)
            root.setOffload(root.offloadIndex)
          }
        }
      }

      // ---------- Audio feedback volume ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        Item {
          width: parent.width
          implicitHeight: Math.max(volumeHeader.implicitHeight, volumePercent.implicitHeight)

          PanelSectionHeader {
            id: volumeHeader
            text: "AUDIO FEEDBACK"
            foreground: root.bar.foreground
            fontFamily: root.bar.fontFamily
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
          }
          Text {
            id: volumePercent
            textFormat: Text.PlainText
            text: root.feedbackVolume + "%"
            color: Qt.darker(root.bar.foreground, 1.4)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.caption
            font.bold: true
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
          }
        }

        PanelSlider {
          anchors.fill: parent
          bar: root.bar
          minimum: 0
          maximum: 100
          step: 1
          integer: true
          value: root.feedbackVolume
          onReleased: function(v) { root.feedbackVolume = Math.round(v); root.setVolume(root.feedbackVolume) }
        }
      }
    }
  }

  Component.onCompleted: root.refresh()
}
