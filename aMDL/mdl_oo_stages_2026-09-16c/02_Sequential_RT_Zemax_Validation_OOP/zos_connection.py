"""
zos_connection.py
-----------------
Reusable connection helper for the Zemax OpticStudio ZOS-API (Python via pythonnet).

Usage in your scripts:

    from zos_connection import PythonStandaloneApplication

    zos = PythonStandaloneApplication()            # headless instance
    zos = PythonStandaloneApplication(mode="extension")   # attach to the
                                                   # OPEN OpticStudio GUI
    TheSystem = zos.TheSystem      # IOpticalSystem
    ZOSAPI = zos.ZOSAPI            # namespace with all enums/types
    ...
    zos.close()                    # standalone: closes the instance;
                                   # extension: only disconnects (the
                                   # GUI and its windows STAY OPEN)

Modes
-----
* mode="standalone" (default): CreateNewApplication() -- a headless
  OpticStudio instance. No GUI; analyses are read via their results
  objects (DataGrids) and exported by the calling script.
* mode="extension": ConnectAsExtension(0) -- attaches to the RUNNING
  OpticStudio GUI. REQUIRED FIRST: in OpticStudio, Programming tab ->
  Interactive Extension (the tile starts "Waiting for connection").
  Every analysis the script opens then appears as a NATIVE OpticStudio
  analysis window in the GUI, with Zemax's own plots, and remains open
  after the script exits -- this is the route for assessing results
  with Zemax's own plotting tools. Note: while the extension is
  connected the GUI shows "an extension has control"; if
  ShowChangesInUI is available it is enabled so updates render live.

This is the standard boilerplate (adapted from the Ansys Zemax sample
code). Requires: pip install pythonnet
"""

import os
import winreg


class PythonStandaloneApplication:
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self, path=None, mode="standalone"):
        # Locate the OpticStudio installation from the Windows registry
        import clr

        aKey = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Zemax", 0, winreg.KEY_READ
        )
        zemaxData = winreg.QueryValueEx(aKey, "ZemaxRoot")
        NetHelper = os.path.join(
            os.sep, zemaxData[0], r"ZOS-API\Libraries\ZOSAPI_NetHelper.dll"
        )
        winreg.CloseKey(aKey)

        clr.AddReference(NetHelper)
        import ZOSAPI_NetHelper

        # Initialize (optionally pass an explicit installation folder)
        if path is None:
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize()
        else:
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(path)

        if not isInitialized:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to locate Zemax OpticStudio. Try specifying the path."
            )

        # Load the ZOS-API assemblies
        clr.AddReference("ZOSAPI")
        clr.AddReference("ZOSAPI_Interfaces")
        import ZOSAPI

        self.ZOSAPI = ZOSAPI
        self.mode = mode
        self.TheConnection = ZOSAPI.ZOSAPI_Connection()
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException(
                "Unable to initialize .NET connection to ZOS-API"
            )

        if mode == "extension":
            # attach to the OPEN GUI: Programming -> Interactive
            # Extension must be waiting, otherwise this returns None
            self.TheApplication = self.TheConnection.ConnectAsExtension(0)
            if self.TheApplication is None:
                raise PythonStandaloneApplication.ConnectionException(
                    "Unable to connect as extension. In OpticStudio: "
                    "Programming tab -> Interactive Extension, then "
                    "re-run this script."
                )
            # render script-driven changes live in the GUI where the
            # API version supports it
            try:
                self.TheApplication.ShowChangesInUI = True
            except Exception:
                pass
        else:
            self.TheApplication = self.TheConnection.CreateNewApplication()
            if self.TheApplication is None:
                raise PythonStandaloneApplication.InitializationException(
                    "Unable to acquire ZOSAPI application"
                )

        if not self.TheApplication.IsValidLicenseForAPI:
            raise PythonStandaloneApplication.LicenseException(
                "License is not valid for ZOS-API use"
            )

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system"
            )

    def close(self):
        """Standalone: close the headless OpticStudio instance.
        Extension: only DISCONNECT -- the GUI, the loaded system and
        every analysis window the script opened stay on screen.

        Call this explicitly (or register it with atexit) when your
        script is done.  It is idempotent.
        """
        if getattr(self, "TheApplication", None) is not None:
            try:
                if getattr(self, "mode", "standalone") == "extension":
                    self.TheApplication.CloseApplication()  # disconnects
                    # the extension; OpticStudio itself remains open
                else:
                    self.TheApplication.CloseApplication()
            except Exception:
                # e.g. the .NET bridge is already torn down at interpreter
                # shutdown -- nothing useful can be done at this point
                pass
            self.TheApplication = None
        self.TheConnection = None

    def __del__(self):
        # Best-effort fallback only: during interpreter shutdown pythonnet
        # may already be unloaded, making .NET calls fail ("'MethodObject'
        # object is not callable").  Prefer an explicit close().
        try:
            self.close()
        except Exception:
            pass

    def samples_dir(self):
        return self.TheApplication.SamplesDir