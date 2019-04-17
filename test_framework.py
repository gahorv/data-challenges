import os
import shutil
import subprocess
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm


def comparison_test(data_dicts,input_set,
                    envdir='env_dir',
                    stagedir='staging',
                    is_windows=False,
                    verbose=False):
    python_path = 'bin/python'
    slash = '/'
    if is_windows:
        python_path = 'Scripts\python.exe'
        slash = '\\'
    if os.path.exists(envdir):
        shutil.rmtree(envdir)
    os.makedirs(envdir)
    solutions = next(os.walk('solutions'))[1]
    solutions.sort()
    solutions = [s for s in solutions if (not s.startswith('.'))]
    timedata = []
    output_set = {}
    for solution in tqdm(solutions):
        print(solution)
        output_set[solution] = {}
        #create venv for all solutions and fill it with the requirements
        solution_env = slash.join([envdir,solution])
        solution_python_executor = slash.join([solution_env,python_path])
        subprocess.call(['python','-m','venv',solution_env])
        subprocess.call([solution_python_executor,'-m',
                         'pip','install','-r','solutions/%s/requirements.txt' % solution])
        
        for data_name,data_dic in tqdm(data_dicts.items()):
            output_set[solution][data_name] = {}
            #create data.json file and run ETL
            data_dic['solution_folder'] = slash.join(['solutions',solution])
            data_dic['solution_python_executor'] = solution_python_executor
            if os.path.exists(stagedir):
                shutil.rmtree(stagedir)
            os.makedirs(stagedir)
            json.dump(data_dic,open('data.json','w'))
            if verbose:
                print('Starting ETL process')
            etl_proc = subprocess.Popen([solution_python_executor,
                            slash.join(['solutions',solution,'ETL.py'])],stderr=subprocess.PIPE)
            if verbose:
                print('ETL process started')
          
            running_time = 0
            while etl_proc.poll() is None:
                time.sleep(3)
                running_time += 3
                if (running_time >= 60) and (running_time % 10) == 0:
                    print('ETL process taking over %d seconds' % running_time)
            if etl_proc.poll() != 0:
                print('error in ETL')
                print(etl_proc.stderr.read().decode('utf-8'))
                raise(RuntimeError)
                
            if verbose:
                print('ETL process done')
            
            for input_name,input_dictlist in input_set[data_name].items():
                output_set[solution][data_name][input_name] = []
                for input_dic in input_dictlist:
                    json.dump(input_dic,open('input.json','w'))
                    start_time = time.time()                    
                    _proc = subprocess.Popen([solution_python_executor,
                            slash.join(['solutions',solution,'process.py'])],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                    _proc.wait()
                    (_stdout, _stderr) = _proc.communicate()
                    if _proc.returncode != 0:
                        print(_stderr.decode('utf-8'))
                        raise(RuntimeError)
                    calc_time = time.time() - start_time
                    timedata.append({'calc_time':calc_time,
                                    'input_id':input_name,
                                    'data_id':data_name,
                                    'solution':solution})
                    #record output
                    output_set[solution][data_name][input_name].append(
                        json.load(open('output.json','r')))
                    #cleanup
                    os.remove('output.json')
                    os.remove('input.json')
            try:
                if verbose:
                    print('Starting cleanup process')
                clean_proc = subprocess.Popen([solution_python_executor,
                                slash.join(['solutions',solution,'cleanup.py'])])
                if verbose:
                    print('cleanup process started')
              
                while clean_proc.poll() is None:
                    time.sleep(3)
                if verbose:
                    print('cleanup process done')
                clean_proc.kill()
            except:
                pass
                if verbose:
                    print('No cleanup process')
            etl_proc.kill()
            etl_proc.terminate()
            if verbose:
                print('ETL killed')
            if os.path.exists(stagedir):
                shutil.rmtree(stagedir)
            os.makedirs(stagedir)
            os.remove('data.json')
    
    shutil.rmtree(stagedir)
    shutil.rmtree(envdir)        
    out_dfs = {}
    for solution,out in output_set.items():
        out_dfs[solution] = pd.DataFrame()
        for data_name,data_out in out.items():
            for input_name,input_out in data_out.items():
                _df = pd.DataFrame([j for jlist in input_out for j in jlist])
                _df['input_id'] = input_name
                _df['data_id'] = data_name
                out_dfs[solution] = pd.concat([out_dfs[solution],_df.copy()])
    return timedata,out_dfs
