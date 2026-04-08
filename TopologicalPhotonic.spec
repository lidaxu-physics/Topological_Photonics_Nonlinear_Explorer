# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Topological Photonic Lattice Explorer
# Build command:  pyinstaller TopologicalPhotonic.spec

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

a = Analysis(
    ['Linear.py'],
    pathex=['.'],
    binaries=[
        (r'C:\Users\22842\anaconda3\Library\bin\mkl_intel_thread.2.dll', '.'),
    ] + collect_dynamic_libs('jaxlib'),
    datas=[
        ('NonLinear.py', '.'),
        ('Linear.py',    '.'),
        ('original.png', '.'),
        ('icon.ico', '.'),
    ] + collect_data_files('matplotlib')
      + collect_data_files('jax')
      + collect_data_files('jaxlib'),
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.figure',
        'matplotlib.patches',
        'matplotlib.font_manager',
        'matplotlib.colors',
        'matplotlib.cm',
        'numpy',
        'scipy',
        'scipy.linalg',
        'jax',
        'jax.numpy',
        'jax.interpreters',
        'jax.interpreters.mlir',
        'jax.interpreters.xla',
        'jax.interpreters.partial_eval',
        'jax.interpreters.batching',
        'jax.interpreters.ad',
        'jax._src',
        'jax._src.lax',
        'jax._src.lax.lax',
        'jax._src.numpy',
        'jax.lib',
        'jaxlib',
        'jaxlib.xla_extension',
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

splash = Splash(
    'original.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(20, 395),
    text_size=13,
    text_color='#c8d0e7',
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TopologicalPhotonicExplorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,            # keep True until confirmed working, then flip to False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)
