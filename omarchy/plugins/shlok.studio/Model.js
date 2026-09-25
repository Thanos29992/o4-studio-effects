// Model.js — helpers for the shlok.studio panel.
// Reads/writes small text files under ~/.local/share/studio-effects/state/.
// Imported into Panel.qml as `Model`.

function boolFromFile(value) {
  if (value === null || value === undefined || value === "") return true
  var s = String(value).trim().toLowerCase()
  return !(s === "off" || s === "false" || s === "0" || s === "disabled")
}

function deviceFromFile(value) {
  var s = String(value == null ? "" : value).trim().toUpperCase()
  if (s === "CPU" || s === "GPU" || s === "NPU") return s
  return "NPU"
}
