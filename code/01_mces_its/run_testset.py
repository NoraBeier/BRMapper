#!/usr/bin/env python3

import sys
import os
import subprocess
from tqdm import tqdm
import concurrent.futures  

#run as run_testset.py aam_ringreactions.csv 10 ring_reactions_its
#./code/01_mces_its/run_testset.py data/aam_ringreactions.csv 10 ring_reactions_its/

dataset_path = sys.argv[1]
threads = int(sys.argv[2])
output = sys.argv[3]

script_dir = os.path.abspath( os.path.dirname( __file__ ) )

def external_call(key, reaction):

    cmd = [script_dir+'/run_mcsplit.py', key + ' ' + reaction, output]  # the external command to run
    timeout_s = 300  # how many seconds to wait 
    
    try:
        p = subprocess.run(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f'Timeout for {cmd} ({timeout_s}s) expired')

elements = []
with open(dataset_path, 'r') as f:
    for line in f:
        if line.strip():
            #key, smiles, _ = line.strip().split(maxsplit=2)
            split = line.strip().split(',', maxsplit=2)
            key = split[0]
            smiles = split[1]
            #key, smiles = line.strip().split(',')
            elements.append( (key, smiles) )

with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
    with tqdm(total=len(elements), desc='Calls', disable=False) as progress:
        work_load = {executor.submit(
                     external_call,
                     key, reaction): i for i, (key, reaction) in enumerate(elements)}

        for future in work_load:
            future.add_done_callback(lambda p: progress.update())
        for future in concurrent.futures.as_completed(work_load):
            None



