# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['cogamer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('.env', '.'),  # Include .env file
        ('copilot.png', '.'),  # Include the icon
        ('gui_utils.py', '.'),  # Include GUI utils
    ],
    hiddenimports=[
        'google.generativeai',
        'mss',
        'PIL',
        'cv2',
        'pyaudio',
        'asyncio',
        'dotenv',
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
    name='cogamer',
    debug=True,  # Enable debug for troubleshooting
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)