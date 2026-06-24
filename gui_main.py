# -*- coding: utf-8 -*-
# GUI entry point — run with: python gui_main.py
import os
import sys

# Make sure SakuraLang's own modules are importable (graph, tools, settings, …)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import main

if __name__ == "__main__":
    main()
