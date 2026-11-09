"""Workspace benchmark adaptors: the suite runner, one module per benchmark, launchers and trace exporters."""

import os
import sys
from pathlib import Path

import ogda

from .artifacts import ONTOLOGY_SOURCE, WORKSPACE

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

# The adaptor stays local while an explicit override selects the core method.
if os.environ.get("OGDA_SOURCE") and ONTOLOGY_SOURCE != Path(__file__).resolve().parents[2]:
    ogda.__path__.insert(0, str(ONTOLOGY_SOURCE / "ogda"))
