# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['gui_utils.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('copilot.png', '.'),  # Include the icon
        ('.env', '.'),  # Include environment file
        ('cogamer.exe', '.'),  # Include the cogamer executable
    ],
    hiddenimports=[
        'tkinter',
        'PIL',
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='cogamer_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='copilot.png'  # Optional: Set the icon for the executable
)