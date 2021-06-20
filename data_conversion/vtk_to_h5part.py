import vtk
import os
import h5py
import numpy as np

# Input folder e.g. from dualsphysics
# Is expected to be in vtk format!
folder = 'dualsphysics/examples/chrono/09_Turbine/CaseTurbine_out/particles/'

out_file = 'Turbine.h5part'

# Specify time interval of the data 
time_interval = range(0, 500, 1)


def read_file(file, data):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.Update()

    output = reader.GetOutput()

    num = output.GetNumberOfPoints()

    xs = np.empty(num, dtype=np.float32)
    ys = np.empty(num, dtype=np.float32)
    zs = np.empty(num, dtype=np.float32)
    for p in range(0, num):
        pt = output.GetPoint(p)
        xs[p] = pt[0]
        ys[p] = pt[1]
        zs[p] = pt[2]

    data['X'].append(xs)
    data['Y'].append(ys)
    data['Z'].append(zs)

    ptdata = output.GetPointData()
    num_attrs = ptdata.GetNumberOfArrays()

    for a in range(0, num_attrs):
        name = ptdata.GetArrayName(a)
        values = np.asarray(ptdata.GetAbstractArray(a))

        if name == 'Vel':
            if 'V' not in data:
                #data['U'] = []
                data['V'] = []
                #data['W'] = []

            #data['U'].append(values[:, 0])
            data['V'].append(values[:, 1])
            #data['W'].append(values[:, 2])

        else:
            continue


def get_step(file):
    beg = file.find('_')
    end = file.find('.')
    return int(file[beg+1:end])


def write_step(files, step, out):
    data = { 'X' : [], 'Y' : [], 'Z' : [] }

    for f in files:
        read_file(f, data)

    key = 'Step#' + str(step) + '/'
    for k, l in data.items():
        if l[0].dtype == np.float32 or l[0].dtype == np.uint32:
            out.create_dataset(key + k, data=np.concatenate(l))


if __name__ == '__main__':
	files = sorted(os.listdir(folder))

	steps = { }

	for f in files:
	    if f.endswith('.vtk'):
		s = get_step(f)

		if s not in steps:
		    steps[s] = []
		steps[s].append(folder + f)

	print(steps)

	with h5py.File(out_file, 'w') as out:
	    for s, fs in steps.items():
		if s in time_interval:
		    print('Writing', s)
		    write_step(fs, s, out)
