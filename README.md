# Image-based Visualization of Large Volumetric Data Using Moments

This is a prototype implementation of the chapter in my PhD thesis (DOI: 10.5445/IR/1000131767).
It does not contain an interactive environment - only a command line tool to generate and render moment images.

### Building the project

The project has been developed and tested with Ubuntu 20.04 and 18.04. It should work on comparable Linux distributions. We do not believe that it will work without changes on Windows or Mac.

You can alternatively use docker (see below), but this does (currently) not support CUDA. Otherwise, make sure all dependencies are installed, see also the Dockerfiles in `docker/` which lists all dependencies. Most importantly:

- HDF5
- NetCDF
- lz4
- CUDA (optional, but recommended)

Then proceed with the build:

- Use cmake to prepare the build:
`mkdir build && cd build && cmake ..`

- Build it: `make`

### Build and run with docker

First build the image:

`docker build -f docker/Dockerfile.run . -t mray_run:latest `

Then start the container:

`docker run -it --mount type=bind,source="absolute_path_to_results_folder",target=/mray/results mray_run:latest`

Inside the container you can execute the pre-configured Marschner-Lobb dataset:

`python3 eval/eval.py eval/marschner_lobb/marschner_lobb.json`

This will take a while since it runs on the CPU. Results are written to the specified `absolute_path_to_results_folder`.

To run other datasets and configuration, these must be copied to the container, as specified in `Dockerfile.run`.

### Getting the data

We have included a ready to use Marschner-Lobb dataset for testing.

Datasets from the paper:
- The Rayleigh-Taylor and Richtmyer-Meshkov datasets are available here: https://klacansky.com/open-scivis-datasets/
	- Download the .raw file(s)
	- Use paraview or another tool to create a .vti file.
	- Convert the vti file to nc: `python3 data_conversion/vti_to_nc_.py created_file.vti data/file.nc`

- The Turbine SPH dataset can be generated using a DualSPHysics example case. In the future, we will try to make a subset of this dataset public.

In either case, the data has to be converted to NetCDF (volumes) or HDF5 (particles) using e.g. the python scripts in `eval/dataset_conversion/`.


### Running the evaluations

The C++ project is not designed for manual usage (although it can certainly be used that way). It loosely depends on a python framework built on top.

The python framework requires the libraries specified in `eval/requirements.txt`, namely: python3, Pillow and numpy.

Use the python file `eval/eval.py` to execute a json configuration, e.g. `eval/marschner_lobb/marschner_lobb.json`.

For example, run the Marschner-Lobb dataset with:
	`python3 eval/eval.py eval/marschner_lobb/marschner_lobb.json`.
	Add `--cuda` to use GPU acceleration.


These json configuration files can be edited by hand and are set to the parameters we used for the paper.

There are some general configurations available in `eval/config.json` that specify the path to the c++ binary and the build command (e.g. make).


### Structure of the project

- `src/`: C++ and CUDA source code
- `test/`: Unit tests (requires googletest)
- `eval/`: Executing the prototype with different datasets and parameters
- `cmake/`: Cmake configuration files

### Most relevant code

- `src/mese/MESE_dynamic.h`: Bounded MESE (Sec. 3.1 and 3.4)
- `src/raymarch_generate.h`: Moment generation (Sec. 3.2)
- `src/raymarch_reconstruct.h`: Rendering/reconstructing a moment image (Sec. 3.3)
- `src/moment_prediction_coding.h`: Our coding scheme (Sec. 3.6.1, 3.6.2)
- `src/measure_coding_error.*`: Determining the quantization curve (Sec. 3.6.3)
- `src/moment_compaction.h`: Determining the number of moments (Sec. 3.5)
