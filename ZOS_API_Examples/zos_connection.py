"""
zos_connection.py
-----------------
Reusable connection helper for the Zemax OpticStudio ZOS-API (Python via pythonnet).

Usage in your scripts:

    from zos_connection import PythonStandaloneApplication

    zos = PythonStandaloneApplication()
    TheSystem = zos.TheSystem      # IOpticalSystem
    ZOSAPI = zos.ZOSAPI            # namespace with all enums/types
    ...
    del zos                        # closes the OpticStudio instance cleanly

This is the standard standalone-mode boilerplate (adapted from the Ansys Zemax
sample code). Requires: pip install pythonnet

For INTERACTIVE EXTENSION mode instead (attach to an open GUI session):
  1. In OpticStudio: Programming tab -> Interactive Extension
  2. Replace CreateNewApplication() below with ConnectAsExtension(0)
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

    def __init__(self, path=None):
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
        self.TheConnection = ZOSAPI.ZOSAPI_Connection()
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException(
                "Unable to initialize .NET connection to ZOS-API"
            )

        # --- STANDALONE mode (headless instance). For interactive extension,
        # --- use: self.TheApplication = self.TheConnection.ConnectAsExtension(0)
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

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None
        self.TheConnection = None

    def samples_dir(self):
        return self.TheApplication.SamplesDir
