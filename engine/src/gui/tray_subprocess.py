"""Standalone GTK3 tray icon process — runs separately to avoid GTK3/4 conflict."""

from __future__ import annotations

import signal

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")

from gi.repository import AyatanaAppIndicator3, Gtk


def main() -> None:
    signal.signal(signal.SIGTERM, lambda *_: Gtk.main_quit())
    signal.signal(signal.SIGINT, lambda *_: Gtk.main_quit())

    indicator = AyatanaAppIndicator3.Indicator.new(
        "linux-studio-effects",
        "camera-video-symbolic",
        AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS,
    )
    indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
    indicator.set_title("Linux Studio Effects")

    menu = Gtk.Menu()

    status_item = Gtk.MenuItem(label="Linux Studio Effects")
    status_item.set_sensitive(False)
    menu.append(status_item)

    menu.append(Gtk.SeparatorMenuItem())

    quit_item = Gtk.MenuItem(label="Quit")
    quit_item.connect("activate", lambda *_: Gtk.main_quit())
    menu.append(quit_item)

    menu.show_all()
    indicator.set_menu(menu)

    Gtk.main()


if __name__ == "__main__":
    main()
