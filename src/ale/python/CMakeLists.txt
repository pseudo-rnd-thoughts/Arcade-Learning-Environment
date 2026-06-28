find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind (installed via pip in build-system.requires)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Use nanobind_add_module for proper scikit-build-core integration.
# FREE_THREADED marks the module Py_MOD_GIL_NOT_USED so it loads without
# re-enabling the GIL on free-threaded CPython builds (3.13t, 3.14t, ...).
nanobind_add_module(_ale_py MODULE FREE_THREADED ale_python_interface.cpp)
if (BUILD_VECTOR_LIB AND BUILD_VECTOR_XLA_LIB)
    # XLA Integration setup
    message(STATUS "Python Executable: '${Python_EXECUTABLE}'")
    message(STATUS "Python_SITELIB=${Python_SITELIB}, Python_SITEARCH=${Python_SITEARCH}")
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "-c"
        "import sys; sys.path.append(r'${Python_SITELIB}'); sys.path.append(r'${Python_SITEARCH}'); from jax import ffi; print(ffi.include_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE XLA_DIR
    )
    message(STATUS "XLA include directory: '${XLA_DIR}'")

    target_sources(_ale_py PRIVATE ale_vector_python_interface.cpp ale_vector_xla_interface.cpp)
    target_include_directories(_ale_py PUBLIC ${XLA_DIR})
    target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_LIB BUILD_VECTOR_XLA_LIB)
    set_target_properties(_ale_py PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # CUDA is optional: when found we build GPU handlers as well, otherwise the
    # XLA targets are registered only against the CPU platform. macOS never has
    # CUDA; on Linux/Windows it is auto-detected unless ALE_DISABLE_CUDA=1 sets
    # BUILD_VECTOR_XLA_GPU_LIB=OFF via the scikit-build override.
    if (BUILD_VECTOR_XLA_GPU_LIB)
        find_package(CUDAToolkit QUIET)
        if (NOT CUDAToolkit_FOUND)
            message(FATAL_ERROR
                "CUDA Toolkit not found, but BUILD_VECTOR_XLA_GPU_LIB=ON.\n"
                "  Either install the CUDA Toolkit (nvcc + cudart headers) from\n"
                "    https://developer.nvidia.com/cuda-downloads\n"
                "  or set ALE_DISABLE_CUDA=1 before running pip install to build\n"
                "  XLA with CPU handlers only:\n"
                "    Linux/macOS:  ALE_DISABLE_CUDA=1 pip install -v .\n"
                "    Windows PS:   $env:ALE_DISABLE_CUDA = \"1\"; pip install -v .")
        endif()
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION} - building XLA GPU handlers")
        target_include_directories(_ale_py PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_compile_definitions(_ale_py PRIVATE ALE_GPU_SUPPORT)
        target_link_libraries(_ale_py PRIVATE CUDA::cudart)
    else()
        message(STATUS "CUDA disabled - building XLA CPU handlers only")
    endif()
elseif (BUILD_VECTOR_LIB)
    target_sources(_ale_py PRIVATE ale_vector_python_interface.cpp)
    target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_LIB)
endif()

target_link_libraries(_ale_py PUBLIC ale ale-lib)

# Install the compiled module
install(TARGETS _ale_py LIBRARY DESTINATION ale_py)

# Install Python source files
install(FILES
    __init__.py
    env.py
    registration.py
    vector_env.py
    py.typed
    DESTINATION ale_py)

# Install stub files if they exist
install(FILES
    __init__.pyi
    DESTINATION ale_py
    OPTIONAL)

# Install roms directory
install(DIRECTORY roms/
    DESTINATION ale_py/roms
    FILES_MATCHING PATTERN "*")

# If we're dynamically loading SDL with Python we'll be building a wheel
# so we should prepare SDL for distribution. Add rpath and copy over
# the dynamic library. auditwheel will take care of ensuring
# cross-platform compatibility on macOS and Linux.
if (SDL_SUPPORT AND SDL_DYNLOAD)
    set_target_properties(_ale_py PROPERTIES
        INSTALL_RPATH_USE_ORIGIN TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        MACOSX_RPATH TRUE
        INSTALL_RPATH
        "$<$<PLATFORM_ID:Darwin>:@loader_path>$<$<PLATFORM_ID:Linux>:\$ORIGIN>")

    # Define our SDL2 distribution library name for dynamic loading
    target_compile_definitions(ale
        PRIVATE SDL2_LIBRARY_NAME="$<TARGET_FILE_NAME:SDL2::SDL2>")
    # Copy over SDL2 dist. library
    add_custom_command(TARGET _ale_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2>
            $<TARGET_FILE_DIR:_ale_py>)
    # Install SDL2 library alongside the module
    install(FILES $<TARGET_FILE:SDL2::SDL2> DESTINATION ale_py)
endif()
