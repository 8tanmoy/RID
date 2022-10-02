#plot3d
#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

kbT = (8.617343E-5) * 300
beta = 1.0 / kbT
f_cvt = 96.485
cv_dim=3

xmin = 0.153 #x is P1OH2
ymin = 0.153 #y is P1O4
zmin = -0.200 #z is asym
xmax = 0.400
ymax = 0.400
zmax = 0.200

#---good
def load_graph(frozen_graph_filename,
               prefix = 'load'):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=prefix,
            op_dict=None,
            producer_op_list=None
        )
    return graph

#--good
def graph_op_name (graph) :
    names = []
    for op in graph.get_operations():
        print (op.name)

#--good
def test_e (sess, xx) :
    graph = sess.graph

    inputs  = graph.get_tensor_by_name ('load/inputs:0')
    o_energy= graph.get_tensor_by_name ('load/o_energy:0')

    zero_ncv = np.zeros ([xx.shape[0], cv_dim])
    data_inputs = np.concatenate ((xx, zero_ncv), axis = 1)
    #data_inputs = xx
    feed_dict_test = {inputs: data_inputs}

    data_ret = sess.run ([o_energy], feed_dict = feed_dict_test)
    return data_ret[0]

#--careful
def value_array (sess, ngrid) :
    xx = np.linspace(xmin, xmax, (ngrid))
    yy = np.linspace(ymin, ymax, (ngrid))
    zz = np.linspace(zmin, zmax, (ngrid))
    deltax  = (xmax - xmin) / (ngrid - 1)
    deltay  = (ymax - ymin) / (ngrid - 1)
    deltaz  = (zmax - zmin) / (ngrid - 1)

    my_grid    = np.zeros((ngrid*ngrid*ngrid, cv_dim))

    for i in range(ngrid):
        for j in range(ngrid):
            for k in range(ngrid):
                my_grid[i*ngrid*ngrid + j*ngrid +k, 0] = xmin + i * deltax
                my_grid[i*ngrid*ngrid + j*ngrid +k, 1] = ymin + j * deltay
                my_grid[i*ngrid*ngrid + j*ngrid +k, 2] = zmin + k * deltaz

    ve = test_e(sess, my_grid)
    dd = -1.0*ve

#    for i in range(len(xx)):
#        print("computing grid: %d" % i)
#        for j in range(len(yy)):
#            for k in range(len(zz)):
#                ve = test_e (sess, np.concatenate((zero_grid + xx[i], zero_grid + yy[j], zero_grid + zz[k], my_grid), axis = 1))
#                dd[i*(ngrid+1)*(ngrid+1) + j*(ngrid+1) + k] = (kbT) * np.log(np.sum(delta3 * np.exp(-beta * ve)))
#
#    dd = dd - np.min(dd)
    
    dd = dd - np.min(dd)
    return xx, yy, zz, dd

def print_array (fname, xx, yy, zz, dd) :
    with open(fname, 'w') as fp :
        lx = len(xx)
        ly = len(yy)
        for ii in range (len(xx)) :
            for jj in range (len(yy)) :
                for kk in range(len(zz)) :
                    fp.write ("%f %f %f %f\n" % (xx[ii], yy[jj], zz[kk], dd[ii*lx*ly + jj*ly +kk]) )
            #fp.write ("\n")

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=["frozen_model.pb"], type=str, nargs="*",
                        help="Frozen model file to import")
    parser.add_argument("-n", "--numb-grid", default=60, type=int,
                        help="The number of data for test")
    parser.add_argument("-o", "--output", default="fe.out", type=str,
                        help="output free energy")
    args = parser.parse_args()

    count = 0
    for ii in args.model :
        graph = load_graph(ii)
        with tf.Session(graph = graph) as sess:
            xx, yy, zz, dd = value_array (sess, args.numb_grid)
            dd *= f_cvt
            if count == 0:
                avg0 = dd
            else :
                avg0 += dd
        count += 1
    avg0 /= float(count)
    avg0 -= np.min(avg0)
    print_array(args.output, xx, yy, zz, avg0)

if __name__ == '__main__':
    _main()
