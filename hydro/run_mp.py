#!/usr/bin/env python3

import os
import re
import shutil
import json
import argparse
import numpy as np
import subprocess as sp
import glob
import logging
import multiprocessing as mp    #tanmoy

max_tasks = 100
iter_format = "%03d"
task_format = "%02d"
log_iter_head = "iter " + iter_format + " task " + task_format + ": "
#tanmoy D1003 based on ala-3
cv_dim = 3

bias_name="00.biasMD"
bias_files=["plumed.dat", "test.std.py"]
bias_plm="plumed.dat"
bias_out_conf="confs/"
bias_out_dist="all.dists.out"     #tanmoy D1003
bias_out_plm="plm.dpfe.out"

res_name="01.resMD"
res_files=["cmpf.sh", "mkres.sh", "plumed.res.templ", "tools"]
res_plm="plumed.res.dat"

train_name="02.train"
train_files=["model.py", "main.py", "freeze.py", "plot.py"]

mol_name="mol"
mol_files=["conf.gro", "grompp.mdp", "topol.top", "dist.ndx", "dist.sh", "index.ndx"] #tanmoy D1003 delete alanine.itp

def make_iter_name (iter_index) :
    return "iter.%03d" % iter_index

def record_iter (record, ii, jj) :
    with open (record, "a") as frec :
        frec.write ("%d %d\n" % (ii, jj))        

def log_iter (task, ii, jj) :
    logging.info ((log_iter_head + "%s") % (ii, jj, task))

def repeat_to_length(string_to_expand, length):
    ret = ""
    for ii in range (length) : 
        ret += string_to_expand
    return ret

def log_task (message) :
    header = repeat_to_length (" ", len(log_iter_head % (0, 0)))
    logging.info (header + message)

def replace (file_name, pattern, subst) :
    file_handel = open (file_name, 'r')
    file_string = file_handel.read ()
    file_handel.close ()
    file_string = ( re.sub (pattern, subst, file_string) )
    file_handel = open (file_name, 'w')
    file_handel.write (file_string)
    file_handel.close ()

def make_grompp_bias (gro_file, nsteps, frame_freq) :
    replace (gro_file, "nsteps.*=.*", "nsteps = %d" % nsteps)
    replace (gro_file, "nstxout.*=.*", "nstxout = %d" % frame_freq)
    replace (gro_file, "nstvout.*=.*", "nstvout = %d" % frame_freq)
    replace (gro_file, "nstfout.*=.*", "nstfout = %d" % frame_freq)
    replace (gro_file, "nstenergy.*=.*", "nstenergy = %d" % frame_freq)    

def make_grompp_res (gro_file, nsteps, frame_freq) :
    replace (gro_file, "nsteps.*=.*", "nsteps = %d" % nsteps)
    replace (gro_file, "nstxout.*=.*", "nstxout = %d" % 0)
    replace (gro_file, "nstvout.*=.*", "nstvout = %d" % 0)
    replace (gro_file, "nstfout.*=.*", "nstfout = %d" % 0)
    replace (gro_file, "nstenergy.*=.*", "nstenergy = %d" % 0)

def copy_file_list (file_list, from_path, to_path) :
    for jj in file_list : 
        if os.path.isfile(from_path + jj) :
            shutil.copy (from_path + jj, to_path)
        elif os.path.isdir(from_path + jj) :
            shutil.copytree (from_path + jj, to_path + jj)

def create_path (path) :
    if os.path.isdir(path) : 
        dirname = os.path.dirname(path)
        counter = 0
        while True :
            bk_dirname = dirname + ".bk%03d" % counter
            if not os.path.isdir(bk_dirname) : 
                shutil.move (dirname, bk_dirname) 
                break
            counter += 1
    os.makedirs (path)

def make_bias (iter_index, 
               json_file, 
               graph_files) :
    graph_files.sort()
    fp = open (json_file, 'r')
    jdata = json.load (fp)
    template_dir = jdata["template_dir"]    
    bias_trust_lvl_1 = jdata["bias_trust_lvl_1"]
    bias_trust_lvl_2 = jdata["bias_trust_lvl_2"]

    iter_name = make_iter_name (iter_index)
    work_path = iter_name + "/" + bias_name + "/"  
    mol_path = template_dir + "/" + mol_name + "/"
    bias_path = template_dir + "/" + bias_name + "/"

    create_path (work_path)

    # copy md ifles
    for ii in mol_files :
        if os.path.exists (work_path + ii) :
            os.remove (work_path + ii)
        shutil.copy (mol_path + ii, work_path)
    # if have prev confout.gro, use as init conf
    if (iter_index > 0) :
        prev_bias_path = make_iter_name (iter_index-1) + "/" + bias_name + "/"
        prev_bias_path = os.path.abspath (prev_bias_path) + "/"
        if os.path.isfile (prev_bias_path + "confout.gro") :
            os.remove (work_path + "conf.gro")
            os.symlink (prev_bias_path + "confout.gro", work_path + "conf.gro")
        log_task (("use conf of iter " + iter_format ) % (iter_index - 1))
    # copy bias file
    for ii in bias_files :
        if os.path.exists (work_path + ii) :
            os.remove (work_path + ii)
        shutil.copy (bias_path + ii, work_path)        
    # copy graph files
    for ii in graph_files :
        file_name = os.path.basename(ii)
        abs_path = os.path.abspath(ii)        
        if os.path.exists (work_path + file_name) :
            os.remove (work_path + file_name)
        os.symlink (abs_path, work_path + file_name)        
    
    # config MD
    nsteps = jdata["bias_nsteps"]
    frame_freq = jdata["bias_frame_freq"]
    mol_conf_file = work_path + "grompp.mdp"
    make_grompp_bias (mol_conf_file, nsteps, frame_freq)
    
    # config plumed
    graph_list=""
    counter=0
    for ii in graph_files :
        file_name = os.path.basename(ii)
        if counter == 0 :
            graph_list="%s" % file_name
        else :
            graph_list="%s,%s" % (graph_list, file_name)
        counter = counter + 1
    plm_conf = work_path + bias_plm
    replace(plm_conf, "MODEL=", ("MODEL=%s" % graph_list))
    replace(plm_conf, "TRUST_LVL_1=", ("TRUST_LVL_1=%f" % bias_trust_lvl_1))
    replace(plm_conf, "TRUST_LVL_2=", ("TRUST_LVL_2=%f" % bias_trust_lvl_2))
    replace(plm_conf, "STRIDE=", ("STRIDE=%d" % frame_freq))
    if len(graph_list) == 0 :
        log_task ("brute force MD without NN acc")
    else :
        log_task ("use NN model(s): " + graph_list)
        log_task ("set trust l1 and l2: %f %f" % (bias_trust_lvl_1, bias_trust_lvl_2))

def run_bias (iter_index,
              json_file) :
    iter_name = make_iter_name (iter_index)
    work_path = iter_name + "/" + bias_name + "/"  

    fp = open (json_file, 'r')
    jdata = json.load (fp)
    gmx_prep = jdata["gmx_prep"]
    gmx_run = jdata["gmx_run"]
    gmx_prep_log = "gmx_grompp.log"
    gmx_run_log = "gmx_mdrun.log"
    graph_files = glob.glob (work_path + "/*.pb")
    if len (graph_files) != 0 :
        gmx_run = gmx_run + " -plumed " + bias_plm
    gmx_prep_cmd = gmx_prep + " &> " + gmx_prep_log
    gmx_run_cmd = gmx_run  + " &> " + gmx_run_log    

    cwd = os.getcwd()
    os.chdir(work_path)    
    log_task (gmx_prep_cmd)
    sp.check_call (gmx_prep_cmd, shell = True)
    log_task (gmx_run_cmd)
    sp.check_call (gmx_run_cmd,  shell = True)    
    os.chdir(cwd)

def post_bias (iter_index, 
               json_file) :                             #tanmoy D1003 gmx_angle, gmx_angle_log and cmd
    iter_name = make_iter_name (iter_index)
    work_path = iter_name + "/" + bias_name + "/"  
    
    fp = open (json_file, 'r')
    jdata = json.load (fp)
    gmx_split = jdata["gmx_split_traj"]
    gmx_split_log = "gmx_split.log"
    gmx_split_cmd = gmx_split + " &> " + gmx_split_log
    gmx_dist = jdata["gmx_dist"]
    gmx_dist_log = "gmx_dist.log"
    gmx_dist_cmd = gmx_dist + " &> " + gmx_dist_log
    
    cwd = os.getcwd()
    os.chdir(work_path)    
    if os.path.isdir ("confs") : 
        shutil.rmtree ("confs")
    os.makedirs ("confs")
    log_task (gmx_split_cmd)
    sp.check_call (gmx_split_cmd,  shell = True)    
    log_task (gmx_dist_cmd)
    sp.check_call (gmx_dist_cmd,  shell = True)    
    os.chdir(cwd)

def make_res (iter_index, 
              json_file) :                              #tanmoy D1003 function has been mod for dist see below
    fp = open (json_file, 'r')
    jdata = json.load (fp)
    template_dir = jdata["template_dir"]    
    conf_start = jdata["conf_start"]
    conf_every = jdata["conf_every"]
    nsteps = jdata["res_nsteps"]
    frame_freq = jdata["res_frame_freq"]
    sel_threshold = jdata["sel_threshold"]
    max_sel = jdata["max_sel"]

    base_path = os.getcwd() + "/"
    iter_name = make_iter_name (iter_index)
    res_path = iter_name + "/" + res_name + "/"
    bias_path = iter_name + "/" + bias_name + "/" 
    os.chdir (bias_path)
    bias_path = os.getcwd() + "/"
    os.chdir (base_path)
    templ_mol_path = template_dir + "/" + mol_name + "/"
    templ_res_path = template_dir + "/" + res_name + "/"

    # sel angles
    ## check if we have graph in bias
    graph_files = glob.glob (bias_path + "/*.pb")
    if len (graph_files) != 0 :
        os.chdir (bias_path)
        dists = np.loadtxt (bias_out_plm)   #tanmoy D1003
        np.savetxt ("dists.forsel.out", dists[:,1:])   #tanmoy D1003
        sel_cmd="python3 test.std.py -m *.pb -t %f -d dists.forsel.out --output sel.out --output-dist sel.dist.out &> sel.log" % sel_threshold
        log_task ("select with threshold %f" % sel_threshold)
        log_task (sel_cmd)
        sp.check_call (sel_cmd, shell = True)
        os.chdir (base_path)
        sel_idx = []
        with open (bias_path + "sel.out") as fp :
            for line in fp : 
                sel_idx += [int(x) for x in line.split()]
        conf_start = 0
        conf_every = 1
    else :        
        sel_idx = range (len(glob.glob (bias_path + bias_out_conf + "conf*gro")))

    sel_idx = np.array (sel_idx, dtype = np.int)
    if len(sel_idx) > max_sel :
        log_task ("sel %d confs that is larger than %d, randomly sel %d out of them" % 
                  (len(sel_idx), max_sel, max_sel))
        np.random.shuffle (sel_idx)
        sel_idx = sel_idx[:max_sel]
        sel_idx = np.sort (sel_idx)
    
    res_dists = np.loadtxt (bias_path + bias_out_dist)      #tanmoy D1003
    res_dists = np.reshape (res_dists, [-1, cv_dim])             #tanmoy D1003 4->cv_dim
    res_dists = res_dists[sel_idx]                          #tanmoy D1003
    res_confs = []
    for ii in sel_idx : 
        res_confs.append (bias_path + bias_out_conf + ("conf%d.gro" % ii))    

    assert (len(res_confs) == res_dists.shape[0]), "number of bias out conf does not match out distance"    #tanmoy D1003
    assert (len(sel_idx) == res_dists.shape[0]), "number of bias out conf does not numb sel"                #tanmoy D1003
    nconf = len(res_confs)
    if nconf == 0 : 
        return False

    sel_list=""
    for ii in range (nconf) : 
        if ii == 0 : sel_list = str(sel_idx[ii])
        else : sel_list += "," + str(sel_idx[ii])
    log_task ("selected %d confs, indexes: %s" % (nconf, sel_list))

    create_path (res_path) 

    for ii in range (conf_start, nconf, conf_every) :
        # print (ii, sel_idx[ii])
        work_path = res_path + ("%06d" % sel_idx[ii]) + "/"
        os.makedirs(work_path)
        copy_file_list (mol_files, templ_mol_path, work_path)
        copy_file_list (res_files, templ_res_path, work_path)
        conf_file = bias_path + bias_out_conf + ("conf%d.gro" % sel_idx[ii])
        os.remove (work_path + "conf.gro")
        os.symlink (conf_file, work_path + "conf.gro")
        os.chdir (work_path) 
        mkres_cmd = "./mkres.sh " + np.array2string(res_dists[ii], formatter={'float_kind':lambda x: "%.6f" % x}).replace("[","").replace("]","")
        log_task (("%06d" % sel_idx[ii]) + ": " + mkres_cmd)
        sp.check_call (mkres_cmd, shell = True)
        os.chdir (base_path)
        mol_conf_file = work_path + "grompp.mdp"
        make_grompp_res (mol_conf_file, nsteps, frame_freq)
        replace (work_path + res_plm, "STRIDE=", "STRIDE=%d" % frame_freq)

    return True

#tanmoy---------------------
#for passing in run_res pool
def foo(work_path, base_path, gmx_prep_cmd, gmx_run_cmd):
    #print(work_path)
    #print(os.getpid())
    work_name = os.path.basename (work_path)
    log_task ("%s: %s" % (work_name, gmx_run_cmd))
    os.chdir(work_path)
    sp.check_call (gmx_prep_cmd, shell = True)
    sp.check_call (gmx_run_cmd,  shell = True)
    os.chdir(base_path)
    return
#end------------------------

def run_res (iter_index,
             json_file) :
    fp = open (json_file, 'r')
    jdata = json.load (fp)
    gmx_prep = jdata["gmx_prep"]
    gmx_run = jdata["gmx_run"]
    gmx_run = gmx_run + " -plumed " + res_plm
    gmx_prep_log = "gmx_grompp.log"
    gmx_run_log = "gmx_mdrun.log"
    gmx_prep_cmd = gmx_prep + " &> " + gmx_prep_log
    gmx_run_cmd = gmx_run  + " &> " + gmx_run_log    
    
    iter_name = make_iter_name (iter_index)
    res_path = iter_name + "/" + res_name + "/"  
    base_path = os.getcwd() + "/"

    all_task = glob.glob(res_path + "/[0-9]*[0-9]")
    all_task.sort()
    #tanmoy---------
    pool = mp.Pool(processes=16)
    for work_path in all_task:
        pool.apply_async(foo, args = (work_path, base_path, gmx_prep_cmd, gmx_run_cmd, ))
        #--everything moved to function foo above
        #print (work_path)
        #print (gmx_run_cmd)
        #print(os.getpid())
        # work_name = os.path.basename (work_path)
        # log_task ("%s: %s" % (work_name, gmx_run_cmd))
        # os.chdir(work_path)    
        # sp.check_call (gmx_prep_cmd, shell = True)
        # sp.check_call (gmx_run_cmd,  shell = True)    
        # os.chdir(base_path)
    pool.close()
    pool.join()
    #pool = mp.Pool(processes=25)
    #for work_path in all_task[25:]:
    #    pool.apply_async(foo, args = (work_path, base_path, gmx_prep_cmd, gmx_run_cmd, ))
    #pool.close()
    #pool.join()
    #end-----------

def post_res (iter_index,
              json_file) :
    iter_name = make_iter_name (iter_index)
    res_path = iter_name + "/" + res_name + "/"  
    base_path = os.getcwd() + "/"
    
    all_task = glob.glob(res_path + "/[0-9]*[0-9]")
    all_task.sort()
    cmpf_cmd = "./cmpf.sh"
    cmpf_log = "cmpf.log"
    cmpf_cmd = cmpf_cmd + " &> " + cmpf_log

    centers = []
    force = []
    ndim = 0
    for work_path in all_task:
        work_name = os.path.basename (work_path)
        log_task ("%s: %s" % (work_name, cmpf_cmd))        
        os.chdir (work_path)
        sp.check_call (cmpf_cmd,  shell = True)
        this_centers = np.loadtxt ('centers.out')
        centers = np.append (centers, this_centers)
        this_force = np.loadtxt ('force.out')
        force = np.append (force, this_force)        
        ndim = this_force.size
        assert (ndim == this_centers.size), "center size is diff to force size in " + work_path
        os.chdir(base_path)        

    centers = np.reshape (centers, [-1, ndim])
    force = np.reshape (force, [-1, ndim])
    data = np.concatenate ((centers, force), axis = 1)
    np.savetxt (res_path + 'data.raw', data, fmt = "%.6e")

    norm_force = np.linalg.norm (force, axis = 1)
    log_task ("min|f| = %e  max|f| = %e  avg|f| = %e" % 
              (np.min(norm_force), np.max(norm_force), np.average(norm_force)))    

def make_train (iter_index, 
                json_file) :
    fp = open (json_file, 'r')
    jdata = json.load (fp)
    template_dir = jdata["template_dir"]    

    iter_name = make_iter_name (iter_index)
    train_path = iter_name + "/" + train_name + "/"  
    data_path = train_path + "data/"
    data_file = train_path + "data/data.raw"
    base_path = os.getcwd() + "/"
    templ_train_path = template_dir + "/" + train_name + "/"

    create_path (train_path) 
    # if os.path.exists (train_path) :
    #     shutil.rmtree(train_path)
    # os.makedirs(train_path)
    os.makedirs(data_path)
    copy_file_list (train_files, templ_train_path, train_path)
    
    # collect data
    log_task ("collect data upto %d" % (iter_index))
    for ii in range (iter_index + 1) : 
        target_path = base_path + make_iter_name (ii) + "/" + res_name + "/"
        # print (target_path)
        target_raw = target_path + "data.raw"
        cmd = "cat " + target_raw + " >> " + data_file
        # print (cmd)
        sp.check_call (cmd, shell = True)

def run_train (iter_index, 
               json_file) :
    fp = open (json_file, 'r')
    jdata = json.load (fp)
    numb_model = jdata["numb_model"]
    iter_name = make_iter_name (iter_index)
    train_path = iter_name + "/" + train_name + "/"  
    base_path = os.getcwd() + "/"

    os.chdir (train_path)
    pb_list = []
    for ii in range (numb_model) :
        train_cmd = "./main.py &> main.%03d.log" % ii
        freez_cmd = "./freeze.py -o graph.%03d.pb &> freeze.%03d.log" % (ii,ii)
        log_task (train_cmd)
        sp.check_call (train_cmd, shell = True)
        #tanmoy-----
        #nn_FE_name = "nn_FE.%03d.dat" % ii
        #FE_plot_name = "FE_plot.%03d.png" %ii
        #os.rename("nn_FE.dat", nn_FE_name)
        #os.rename("myfig12.png", FE_plot_name)
        #print("FE & FE_plot saved")
        #end--------
        log_task (freez_cmd)
        sp.check_call (freez_cmd, shell = True)
    os.chdir (base_path)


def run_iter (json_file, init_model) :
    prev_model = init_model
    fp = open (json_file, 'r')
    jdata = json.load (fp)
    numb_iter = jdata["numb_iter"]
    numb_task = 8
    record = "record.abnn"

    iter_rec = [0, -1]
    if os.path.isfile (record) :
        with open (record) as frec :
            for line in frec : 
                iter_rec = [int(x) for x in line.split()]
        logging.info ("continue from iter %03d task %02d" % (iter_rec[0], iter_rec[1]))

    for ii in range (numb_iter) : #numb_iter tanmoy warning
        if ii > 0 :
            prev_model = glob.glob (make_iter_name(ii-1) + "/" + train_name + "/*pb")
        for jj in range (numb_task) : #numb_task tanmoy warning
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1] : 
                continue
            if   jj == 0 :
                log_iter ("make_bias", ii, jj)
                # logging.info ("use prev model " + str(prev_model))
                make_bias (ii, json_file, prev_model)            
            elif jj == 1 :
                log_iter ("run_bias", ii, jj)
                run_bias  (ii, json_file)
            elif jj == 2 :
                log_iter ("post_bias", ii, jj)
                post_bias (ii, json_file)
            elif jj == 3 :
                log_iter ("make_res", ii, jj)
                cont = make_res (ii, json_file)
                if not cont : 
                    log_iter ("no more conf needed at %03d task %02d" % (ii, jj))
                    record_iter (record, ii, jj)
                    return
            elif jj == 4 :
                log_iter ("run_res", ii, jj)
                run_res   (ii, json_file)
            elif jj == 5 :
                log_iter ("post_res", ii, jj)
                post_res  (ii, json_file)
            elif jj == 6 :
                log_iter ("make_train", ii, jj)
                make_train(ii, json_file)
            elif jj == 7 :
                log_iter ("run_train", ii, jj)
                run_train (ii, json_file)                
            else :
                raise RuntimeError ("unknow task %d, something wrong" % jj)

            record_iter (record, ii, jj)

    # make_bias (0, json_file, prev_model)
    # run_bias  (0, json_file)
    # post_bias (0, json_file)
    # make_res  (0, json_file)
    # run_res   (0, json_file)
    # post_res  (0, json_file)
    # make_train(0, json_file)
    # run_train (0, json_file)

    # model_0 = glob.glob (make_iter_name(0) + "/" + train_name + "/*pb")
    # make_bias (1, json_file, model_0)
    # run_bias (1, json_file)
    # post_bias (1, json_file)
    # make_res (1, json_file)
    # run_res (1, json_file)
    # post_res (1, json_file)
    # make_train (1, json_file)
    # run_train (1, json_file)
    # prev_model = glob.glob (make_iter_name(1) + "/" + train_name + "/*pb")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("JSON", type=str, 
                        help="The json parameter")
    parser.add_argument("-m", "--models", default=[], nargs = '*', type=str, 
                        help="The init guessed model")    
    args = parser.parse_args()

    logging.basicConfig (level=logging.INFO, format='%(asctime)s %(message)s')
    # logging.basicConfig (filename="compute_string.log", filemode="a", level=logging.INFO, format='%(asctime)s %(message)s')

    logging.info ("start running")
    run_iter (args.JSON, args.models)

