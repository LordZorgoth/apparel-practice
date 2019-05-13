# IMPORTANT NOTE: This script as currently implemented
# only works with functions defined at the top level.
# In particular, it DOES NOT WORK with functions defined as part of a class!
# This script also assumes that docstrings are properly indented, and
# that you don't try to use implicit continuation after a colon.

import os
import glob
import subprocess


def is_tracked_by_git(file_name):
    os.chdir('..')
    proc=subprocess.Popen(['git', 'ls-files', 'code/' + file_name],
                          stdout=subprocess.PIPE)
    return_value=bool(proc.stdout.read())
    os.chdir('code')
    return return_value


os.chdir('../../code')
python_files = glob.glob('**.py', recursive=True)
python_files=[file_name for file_name in python_files
              if is_tracked_by_git(file_name)]
python_files=[os.path.relpath(file_name, '../docs/scripts')
              for file_name in python_files]
directories = glob.glob('**', recursive=True)
os.chdir('../docs/scripts')
docstrings=[]
for file_name in python_files:
    docstring_found = True
    with open(file_name, 'r') as f:
        # I find this loop very ugly, but I don't see a better way to do it.
        function_name=''
        while True:
            line = f.readline()
            if line[:4] == 'def ':
                function_name = line[4:line.find('(')]
                function_info = line[4:].strip()
                end_of_def_found=False
                module_name = file_name[11:-3]
                if function_name and not docstring_found:
                    function_info = line[4:].strip()
                    docstrings.append([module_name,function_name,
                                       function_info
                                     + "\nThis function is undocumented."])
                docstring_found = False
                docstring_done = False
            elif not end_of_def_found:
                
                if '"""' in line:
                    docstring = ''
                    docstring_found = True
                    split_line=line.split('"""')
                    if len(split_line) > 2:
                        docstring_done = True
                    docstring += split_line[1].strip()
                    while not docstring_done:
                        line=f.readline()
                        if '"""' in line:
                            docstring += line.split('"""')[0][4:]
                            docstrings.append([module_name,
                                               function_name,
                                               docstring.strip()])
                            print("Docstring recorded for "
                                  +module_name+"."+function_name)
                            docstring_done = True
                        elif line == '':
                            print('Bad Python module '
                                  + module_name
                                  + ': docstring does not terminate!')
                            break
                        else:
                            if line.strip()!='':
                                docstring += line[4:]
                            else:
                                docstring += '\n'
            if line == '':
                break
module_name=''
os.chdir('../docstrings')
for docstring in docstrings:
    old_module_name = module_name
    module_name = docstring[0]
    if module_name != old_module_name:
        if not os.path.exists(module_name):
            os.mkdir(module_name)
    function_name = docstring[1]
    with open(module_name + '/' + function_name, 'w') as f:
        f.write(docstring[2])
os.chdir('../scripts')
