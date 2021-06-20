import vtk
from vtk.util import numpy_support
import numpy as np
import argparse
from netCDF4 import Dataset
import os


def convert_file(input, output):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(args.input)
    reader.Update()

    imgdata = reader.GetOutput()
    dims = imgdata.GetDimensions()

    bounds = imgdata.GetBounds()

    with Dataset(args.output, 'w') as nc:
        nc.createDimension('xdim', dims[0])
        nc.createDimension('ydim', dims[1])
        nc.createDimension('zdim', dims[2])

        nc.spatial_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
        nc.spatial_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
        nc.time_domain = np.array([0.0, 1.0], dtype=np.float32)

        for sidx in range(0, imgdata.GetScalarSize()):
            array = imgdata.GetPointData().GetArray(sidx)
            if array == None:
                continue

            var = array.GetName()

            field = numpy_support.vtk_to_numpy(array).reshape((dims[2], dims[1], dims[0]))
            del reader
            
            nc.createVariable(var, field.dtype, ('zdim', 'ydim', 'xdim'))

            if field.dtype != np.uint8:
                min = np.min(field)
                max = np.max(field)

                nc[var][:] = (field - min) / (max - min)

            else:
                for z in range(0, dims[2]):
                    nc[var][z, :] = field[z, :]
                    print(z)


def convert_dir(input_dir, output):
    files = sorted(os.listdir(input_dir))[0:50:5]

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_dir + '/' + files[0])
    reader.Update()
    imgdata = reader.GetOutput()
    dims = imgdata.GetDimensions()
    bounds = imgdata.GetBounds()

    scalar_ids = []
    minima = []
    maxima = []

    with Dataset(output, 'w') as nc:
        nc.createDimension('xdim', dims[0])
        nc.createDimension('ydim', dims[1])
        nc.createDimension('zdim', dims[2])
        nc.createDimension('tdim', len(files))

        nc.spatial_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
        nc.spatial_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
        nc.time_domain = np.array([0.0, len(files)], dtype=np.float32)

        for sidx in range(0, imgdata.GetScalarSize()):
            array = imgdata.GetPointData().GetArray(sidx)

            if array == None or array.GetDataType() != vtk.VTK_FLOAT:
                continue

            name = array.GetName()
            nc.createVariable(name, np.float32, ('tdim', 'zdim', 'ydim', 'xdim'))

            field = numpy_support.vtk_to_numpy(array).reshape((dims[2], dims[1], dims[0]))
            minima.append(np.min(field))
            maxima.append(np.max(field))

            scalar_ids.append(sidx)

        print('Finding global min/max')
        for t, f in enumerate(files):
            reader.SetFileName(input_dir + '/' + f)
            reader.Update()
            imgdata = reader.GetOutput()

            for i, sidx in enumerate(scalar_ids):
                field = numpy_support.vtk_to_numpy(imgdata.GetPointData().GetArray(sidx)).reshape((dims[2], dims[1], dims[0]))
                minima[i] = np.minimum(minima[i], (np.min(field)))
                maxima[i] = np.maximum(maxima[i], (np.max(field)))

        print('Writing')
        for t, f in enumerate(files):
            reader.SetFileName(input_dir + '/' + f)
            reader.Update()
            imgdata = reader.GetOutput()

            for i, sidx in enumerate(scalar_ids):
                array = imgdata.GetPointData().GetArray(sidx)
                var = array.GetName()
                field = numpy_support.vtk_to_numpy(array).reshape((dims[2], dims[1], dims[0]))

                nc[var][t, :] = (field - minima[i]) / (maxima[i] - minima[i])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts VTK VTI to the NetCDF format')
    parser.add_argument('input', type=str, help='input vti file or directory')
    parser.add_argument('output', type=str, help='output nc file')
    args = parser.parse_args()

    if args.input.endswith('.vti') and os.path.isfile(args.input):
        convert_file(args.input, args.output)
    elif os.path.isdir(args.input):
        convert_dir(args.input, args.output)
    else:
        print("Not a valid .vti file or a directory!")

