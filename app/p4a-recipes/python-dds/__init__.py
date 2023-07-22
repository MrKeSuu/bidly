import os
import sh

from pythonforandroid.recipe import Recipe, IncludedFilesBehaviour
from pythonforandroid.util import current_directory
from pythonforandroid.logger import shprint, info


ARCH_ALIAS_MAP = {
    'arm64-v8a': 'aarch64',
    'armeabi-v7a': 'arm',
}


class PythonDdsRecipe(IncludedFilesBehaviour, Recipe):

    version = '0.1.0'

    # depends = ['libiconv']

    src_filename = 'dds'
    need_stl_shared = True

    built_libraries = {'libdds.so': 'src'}

    def get_recipe_env(self, arch=None, with_flags_in_cc=True): # TODO
        env = super().get_recipe_env(arch, with_flags_in_cc)
        # libiconv = self.get_recipe('libiconv', self.ctx)
        # libiconv_dir = libiconv.get_build_dir(arch.arch)
        # env['CFLAGS'] += ' -I' + os.path.join(libiconv_dir, 'include')
        # env['LIBS'] = env.get('LIBS', '') + ' -landroid -liconv'
        return env

    def build_arch(self, arch):
        env = self.get_recipe_env(arch)
        makefile_dir = os.path.join(self.get_build_dir(arch.arch), 'src')
        with current_directory(makefile_dir):
            shprint(sh.make, '-f', 'Makefile_Android_aarch64_shared', _env=env)

    def postbuild_arch(self, arch):
        super().postbuild_arch(arch)

        # OpenMP
        arch_alias = ARCH_ALIAS_MAP.get(arch.arch, arch.arch)
        relpath = f'toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.6/lib/linux/{arch_alias}/libomp.so'
        libomp = os.path.join(self.ctx.ndk_dir, relpath)

        info("Installing OpenMP lib manually for %s: %s", arch_alias, libomp)
        self.install_libs(arch, libomp)


recipe = PythonDdsRecipe()
