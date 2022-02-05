# ##### BEGIN GPL LICENSE BLOCK #####
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####

import bpy
import os, sys
import subprocess
from pathlib import Path

# THIRD_PARTY = Path(__file__).resolve().parent / "libs"# os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
# DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"# os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
# SUBPROCESS_DIR = PYTHON_PATH.parent

PYTHON_PATH = Path(sys.executable)
BLENDER_SITE_PACKAGE = PYTHON_PATH.parents[1] / 'lib' / 'site-packages'
# or: BLENDER_SITE_PACKAGE = Path(bpy.utils.resource_path('LOCAL')) / 'python' / 'lib' / 'site-packages'


def module_can_be_imported(name):
    try:
        __import__(name)
        return True
    except ModuleNotFoundError:
        return False

def install_pip():
    # pip can not necessarily be imported into Blender after this
    subprocess.run([str(PYTHON_PATH), "-m", "ensurepip"])

def install_package(name):
    # import logging
    # logging.debug(f"Using {PYTHON_PATH} for installation")
    ## !!! try with , '--user' ?
    subprocess.run([str(PYTHON_PATH), "-m", "pip", "install", name])

def setup(dependencies):
    '''
    Get a set containing multiple tuple of (module, package) names pair (str)
    install pip with ensurepip if needed, try to import, install, retry
    '''

    os.environ['PYTHONUSERBASE'] = str(BLENDER_SITE_PACKAGE)
    if not module_can_be_imported("pip"):
        install_pip()

    for module_name, package_name in dependencies:
        if not module_can_be_imported(module_name):
            print(f'Installing module: {module_name} - package: {package_name}')
            install_package(package_name)
            module_can_be_imported(package_name)


### --- classic install (not used)

def pip_install(package_name):
    '''Get a package name (str) and try to install and print in console'''
    print(f'---Installing {package_name}---')
    try:
        output = subprocess.check_output([bpy.app.binary_path_python, '-m', 'pip', 'install', package_name])
        print(output)
    except subprocess.CalledProcessError as e:
        print(e.output)
        return e.output


def pip_install_and_import(dependencies):
    '''
    Get a set containing multiple tuple of (module, package) names pair (str)
    try to import, if import fail, try to install then try to reimport
    '''
    # os.environ['PYTHONUSERBASE'] = str(BLENDER_SITE_PACKAGE)
    for module_name, package_name in dependencies:
        # '--user'# install for all version of blender (suposely in app data roaming or config files...)
        # '--no-deps'# dont update dependancy (in this case, avoid installing downloading a duplication of  numpy)

        try:
            __import__(module_name)
            continue

        except ImportError:
            try:
                ## auto install dependancy (need to run as admin)
                print(f'Installing module: {module_name} - package: {package_name}') #"--user" ?

                #### Using target
                ## within built-in modules (need run as admin in most case)
                # subprocess.check_call([str(PYTHON_PATH), "-m", "pip", "install", f'--target={BLENDER_SITE_PACKAGE}', package_name])
                
                done = False
                
                ## within external modules (if script files are there)
                external_scripts = bpy.context.preferences.filepaths.script_directory
                print('external_scripts: ', external_scripts)
                print('__file__: ', __file__)
                in_external = str(Path(__file__)).startswith(str(Path(external_scripts)))
                
                if external_scripts and len(external_scripts) > 2 and in_external:
                    external_scripts = Path(external_scripts)
                    if external_scripts.exists():
                        external_modules = external_scripts / 'modules'
                        print(f'using external scripts modules: {external_modules}')
                        external_modules.mkdir(exist_ok=True)# dont raise error if already exists
                        
                        # cmd = [str(PYTHON_PATH), "-m", "pip", "install", f'--target={external_modules}', package_name]
                        cmd = [str(PYTHON_PATH), "-m", "pip", '--no-cache-dir', "install", f'--target={external_modules}', package_name, '--no-deps']
                        print('Run', ' '.join(cmd))
                        subprocess.check_call(cmd)
                        done=True
                
                ## within user local modules (if not in external scripts)
                if not done:
                    user_module = Path(bpy.utils.user_resource('SCRIPTS', path='modules', create=True)) # create the folder if not exists
                    print(f'Using users modules: {user_module}')
                    
                    # cmd = [str(PYTHON_PATH), "-m", "pip", "install", f'--target={user_module}', package_name]
                    cmd = [str(PYTHON_PATH), "-m", "pip", '--no-cache-dir', "install", f'--target={user_module}', package_name, '--no-deps']
                    print('Run', ' '.join(cmd))
                    subprocess.check_call(cmd)

            except Exception as e:
                print(f'{package_name} install error: {e}')
                print('Maybe try restarting blender as Admin')
                return e
        
        try:
            __import__(module_name)
        except ImportError as e:
            print(f'!!! module {module_name} still cannot be imported')
            return e


"""
## addons Paths

# natives
built_in_addons = os.path.join(bpy.utils.resource_path('LOCAL') , path='scripts', 'addons')

# users
users_addons = bpy.utils.user_resource('SCRIPTS', path='addons')

#external
external_addons = None
external_script_dir = bpy.context.preferences.filepaths.script_directory
if external_script_dir and len(external_script_dir) > 2:
    external_addons = os.path.join(external_script_dir, 'addons')
"""
            