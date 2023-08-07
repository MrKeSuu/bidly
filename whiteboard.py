# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: bidly
#     language: python
#     name: bidly
# ---

# %%
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

# %%
import logging
import sys

import detect
import solve


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# %%
image_handler = detect.get_image_handler()
image_handler.read('../deal5-md-sq.jpg')
image_handler.validate()
image_input = image_handler.preprocess()

# %%
ONNX_MODEL_PATH = '../detector/best.onnx'
ONNX_MODEL_PATH = '../detector/best.onnx'

yolo5 = detect.Yolo5Opencv(detect.OpencvOnnxLoader())
yolo5.load(ONNX_MODEL_PATH)

# %%
detection = yolo5.detect(image_input)
len(detection)

# %%
detection[:3]

# %%
solver = solve.BridgeSolver(detection, presenter=solve.PrintPresenter())

# %%
solver.transform()
solver.assign()
solver.solve()
solver.present()

# %%

# %%

# %% [markdown]
# ## P4A

# %%
env = {'HOME': '/home/yiqian', 'CFLAGS': '-target aarch64-linux-android21 -fomit-frame-pointer -march=armv8-a -fPIC', 'CXXFLAGS': '-target aarch64-linux-android21 -fomit-frame-pointer -march=armv8-a -fPIC', 'CPPFLAGS': '-DANDROID -I/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include -I/home/yiqian/repos/bidly/.buildozer/android/platform/build-arm64-v8a/build/python-installs/bidly/arm64-v8a/include/python3.8', 'LDFLAGS': '  -L/home/yiqian/repos/bidly/.buildozer/android/platform/build-arm64-v8a/build/libs_collections/bidly/arm64-v8a', 'LDLIBS': '-lm', 'PATH': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin:/home/yiqian/.buildozer/android/platform/android-ndk-r25b:/home/yiqian/.buildozer/android/platform/android-sdk/tools:/home/yiqian/.buildozer/android/platform/apache-ant-1.9.4/bin:/home/yiqian/repos/bidly/kivyvenv/bin:/usr/local/cuda-12.0/bin:/home/yiqian/miniconda3/condabin:/home/yiqian/.pyenv/shims:/home/yiqian/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin', 'CC': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang -target aarch64-linux-android21 -fomit-frame-pointer -march=armv8-a -fPIC', 'CXX': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ -target aarch64-linux-android21 -fomit-frame-pointer -march=armv8-a -fPIC', 'AR': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar', 'RANLIB': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ranlib', 'STRIP': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip --strip-unneeded', 'READELF': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf', 'OBJCOPY': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objcopy', 'MAKE': 'make -j32', 'ARCH': 'arm64-v8a', 'NDK_API': 'android-21', 'LDSHARED': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang -target aarch64-linux-android21 -fomit-frame-pointer -march=armv8-a -fPIC -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions', 'BUILDLIB_PATH': '/home/yiqian/repos/bidly/.buildozer/android/platform/build-arm64-v8a/build/other_builds/hostpython3/desktop/hostpython3/native-build/build/lib.linux-x86_64-3.8'}
len(env)

# %%
env

# %%

# %%

# %%
arch = {'arch': 'arm64-v8a', 'arch_cflags': ['-march=armv8-a', '-fPIC'], 'clang_exe': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang', 'clang_exe_cxx': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++', 'command_prefix': 'aarch64-linux-android', 'common_cflags': ['-target {target}', '-fomit-frame-pointer'], 'common_cppflags': ['-DANDROID', '-I{ctx.ndk.sysroot_include_dir}', '-I{python_includes}'], 'common_ldflags': ['-L{ctx_libs_dir}'], 'common_ldlibs': ['-lm'], 'common_ldshared': ['-pthread', '-shared', '-Wl,-O1', '-Wl,-Bsymbolic-functions'], 'extra_global_link_paths': [], 'include_dirs': [], 'ndk_lib_dir': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android', 'ndk_lib_dir_versioned': '/home/yiqian/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/21', 'target': 'aarch64-linux-android21'}
len(arch)

# %%
arch

# %%
