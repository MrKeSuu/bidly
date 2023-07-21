import os
import sh

from pythonforandroid.recipe import Recipe, IncludedFilesBehaviour
from pythonforandroid.util import current_directory
from pythonforandroid.logger import shprint


class PythonDdsRecipe(IncludedFilesBehaviour, Recipe):

    version = '0.1.0'

    # depends = ['libiconv']

    src_filename = 'dds'

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


recipe = PythonDdsRecipe()
