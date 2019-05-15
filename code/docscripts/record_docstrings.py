# IMPORTANT NOTE: This function as currently implemented
# only works with functions defined at the top level.
# It DOES NOT WORK with functions defined as part of a class!
# This function also assumes that the pep8 style guide is followed,
# and that you don't try to use implicit continuation after a colon.

import os
import glob
import subprocess


def is_tracked_by_git(file_name):
    os.chdir('..')
    proc = subprocess.Popen(['git', 'ls-files', 'code/' + file_name],
                            stdout=subprocess.PIPE)
    return_value = bool(proc.stdout.read())
    os.chdir('code')
    return return_value


os.chdir('..')
python_files = glob.glob('**.py', recursive=True)
python_files = [file_name for file_name in python_files
                if is_tracked_by_git(file_name)]
python_files = [os.path.relpath(file_name, './docscripts')
                for file_name in python_files]
directories = glob.glob('**', recursive=True)
os.chdir('docscripts')
docstrings = []
for file_name in python_files:
    with open(file_name, 'r') as f:
        module_name = file_name[11:-3]
        function_name = ''
        line = True
        while line:
            line = f.readline()
            if line[:4] == 'def ':
                function_name = line[4:line.find('(')]
                function_info = line[4:].rstrip()+'\n'
                if function_info[-2] == ':':
                    end_of_def_found = True
                    function_info = function_info[:-2]+'\n'
                else:
                    end_of_def_found = False
                while not end_of_def_found:
                    line = f.readline().rstrip()
                    if line[-1] == ':':
                        end_of_def_found = True
                        function_info += line[4:-1]+'\n'
                    elif line == '':
                        print('Corrupted module ' + module_name + '.py:'
                              + 'End of paragraph in function '
                              + 'definition, or function definition ended '
                              + 'mid-line.\nDo not trust script output.')
                        break
                    else:
                        function_info += line[4:]+'\n'
                line = f.readline()
                if '"""' in line:
                    docstring = ''
                    split_line = line.split('"""')
                    if len(split_line) > 2:
                        docstring_done = True
                    else:
                        docstring_done = False
                    docstring += split_line[1].strip()
                    while not docstring_done:
                        line = f.readline()
                        if '"""' in line:
                            docstrings.append([module_name,
                                               function_name,
                                               docstring.strip()])
                            print("Docstring recorded for "
                                  + module_name+"."+function_name)
                            docstring_done = True
                        elif line == '':
                            print('Bad Python module '
                                  + module_name
                                  + ': docstring does not terminate!')
                            break
                        else:
                            if line.strip() != '':
                                docstring += line[4:]
                            else:
                                docstring += '\n'
                else:
                    docstrings.append([module_name, function_name,
                                       function_info
                                       + "\nThis function is undocumented."])
                    print("Function " + module_name + "." + function_name
                          + " is undocumented.")
module_name = ''
os.chdir('../../docs/docstrings')
for docstring in docstrings:
    old_module_name = module_name
    module_name = docstring[0]
    if module_name != old_module_name:
        if not os.path.exists(module_name):
            os.mkdir(module_name)
    function_name = docstring[1]
    with open(module_name + '/' + function_name, 'w') as f:
        f.write(docstring[2])
os.chdir('../../code/docscripts')
