# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Topological Photonic Lattice Explorer
# Build command:  pyinstaller TopologicalPhotonic.spec

block_cipher = None

a = Analysis(
    ['Linear.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('NonLinear.py', '.'),   # include as data so importlib can load it at runtime
        ('original.png', '.'),   # splash / background image
        ('icon.ico', '.'),       # window icon
    ],
    hiddenimports=[
        # Qt / Matplotlib back-ends
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
        'matplotlib.patches',
        # Scientific stack
        'numpy',
        'scipy',
        'scipy.linalg',
        # JAX (optional – included so it works if installed)
        'jax',
        'jax.numpy',
        'jax.interpreters',
        'jax.lib',
        'jaxlib',
        # std-lib
        'importlib.util',
        'concurrent.futures',
        'threading',
        'queue',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'PySide2'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── Splash screen (shown instantly on launch, closed when app is ready) ────────
splash = Splash(
    'original.png',          # your image
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(20, 380),      # position of the loading text (x, y from top-left)
    text_size=13,
    text_color='white',
    text_default='Loading…',
    minify_script=True,
    always_on_top=True,
    full_tk=False,           # minimal Tk — hides the file-list scrolling
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    splash,                  # bundle splash into the exe
    splash.binaries,
    [],
    name='TopologicalPhotonicExplorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # --windowed: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',         # application icon
)
