Versions of cuDNN before the 8.0 release series do not support the NVIDIA Ampere Architecture
and will generate incorrect results if used on that architecture. 
Furthermore, if used, training operations can succeed with a NaN loss for every epoch. 


[adidishe@spartan-login1 ~]$ module avail 

---- /apps/easybuild-2022/easybuild/modules/all/MPI/GCC/11.3.0/OpenMPI/4.1.4 ----
   ABySS/2.2.5
   AFNI/23.1.07
   ANTs/2.4.4
   ARAGORN/1.2.41
   AUGUSTUS/3.5.0
   AmberTools/22.3
   Arrow/8.0.0
   BAMM/2.5.0
   BLAST+/2.13.0
   BUSCO/5.4.5
   Biopython/1.79
   Bottleneck/1.3.7
   CDO/2.2.0
   CGAL/4.14.3
   CNVnator/0.3.3
   CPMD/4.3
   CloudCompare/2.12.4
   Drishti/3.1
   ELPA/2021.11.001
   ELPA/2022.05.001                             (D)
   EMBOSS/6.6.0
   ESMF/8.3.0
   FDS/6.7.9
   FFTW.MPI/3.3.10                              (L)
   FLASH/2.2.00
   GDAL/3.5.0
   GRASP/2018
   GROMACS/2023.1
   GROMACS/2023.2-CUDA-11.7.0                   (D)
   GraphicsMagick/1.3.36
   HDF5/1.12.2
   HISAT2/2.2.1
   HMMER/3.3.2
   HTSeq/2.0.2
   HyPhy/2.5.33
   Hypre/2.25.0
   IQ-TREE/2.2.2.6
   ITK/5.2.1
   Infernal/1.1.4
   InterProScan/5.55-88.0
   JAGS/4.3.1
   KaHIP/3.14
   Kraken2/2.1.2
   LAMMPS/23Jun2022-kokkos-CUDA-11.7.0
   LASTZ/1.04.03
   MACS2/2.2.8
   MEME/5.5.3
   MINC/2.4.03
   MMseqs2/14-7e284
   MRtrix/3.0.4
   MUMPS/5.5.1-metis
   MaxBin/2.2.7
   MultiQC/1.14
   Multiwell/2023.1-u
   NAMD/2.14-CUDA-11.7.0
   NCO/5.1.3
   NLTK/3.7
   Nektar++/5.3.0
   ORCA/5.0.4
   Octave/7.1.0
   OpenCV/4.6.0-contrib
   OpenFOAM/v2206
   PETSc/3.17.4
   PETSc/3.18.3
   PETSc/3.19.2                                 (D)
   PICI-LIGGGHTS/3.8.1
   PLUMED/2.8.1
   ParMETIS/4.0.3
   ParaView/5.10.1-mpi
   PnetCDF/1.12.3
   PyMOL/2.5.0
   PyTorch/1.12.0
   PyTorch/1.12.1-CUDA-11.7.0
   PyTorch/1.12.1                               (D)
   QUAST/5.2.0
   QuantumESPRESSO/7.1
   R-bundle-Bioconductor/3.16-R-4.2.2
   R/4.2.1-bare
   R/4.2.1
   R/4.2.2                                      (D)
   RASPA2/2.0.47
   RAxML/8.2.12-hybrid-avx2
   ROOT/6.26.10
   RSEM/1.3.3
   RStudio-Server/2022.07.2+576-Java-11-R-4.2.1
   SCOTCH/7.0.1
   SEPP/4.5.1
   SKESA/2.4.0_saute.1.3.0_1
   SLEPc/3.17.2
   SLEPc/3.18.3                                 (D)
   SRA-Toolkit/3.0.3
   SRA-Toolkit/3.0.5                            (D)
   SUNDIALS/6.3.0
   ScaFaCoS/1.0.1
   ScaLAPACK/2.2.0-fb                           (L)
   SciPy-bundle/2022.05test
   SciPy-bundle/2022.05                         (D)
   Seaborn/0.12.1
   SuiteSparse/5.13.0-METIS-5.1.0
   SuperLU_DIST/8.1.0
   TensorFlow/2.11.0-CUDA-11.7.0
   TensorFlow/2.11.0                            (D)
   TopHat/2.1.2
   Trinity/2.15.1
   Trinotate/4.0.1
   US-Align/20221221
   VMD/1.9.4a57
   VTK/9.2.2
   Valgrind/3.20.0
   arpack-ng/3.8.0
   arrow-R/8.0.0-R-4.2.2
   barrnap/0.9
   bokeh/2.4.3
   dask/2022.10.0
   deepTools/3.5.2
   dorado/0.3.1-CUDA-11.7.0
   eagle/1.1.3
   ecCodes/2.27.0
   eggnog-mapper/2.1.9
   freebayes/1.3.6
   gmsh/4.11.1
   h5py/3.7.0
   imkl/2022.1.0                                (D)
   kallisto/0.48.0
   libxc/5.2.3                                  (D)
   magma/2.6.2-CUDA-11.7.0
   matplotlib/3.5.2
   mpi4py/3.1.4
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
lines 131-153
   nanopolish/0.13.3
   ncbi-vdb/2.11.2
   ncbi-vdb/3.0.2
   ncbi-vdb/3.0.5                               (D)
   ncdf4/1.17-R-4.2.2
   ncview/2.1.8
   netCDF-C++4/4.3.1
   netCDF-Fortran/4.6.0
   netCDF/4.9.0
   networkx/2.8.4
   numba/0.56.4
   numexpr/2.8.4
   occt/7.5.0p1
   parsnp/1.5.3
   prokka/1.14.5
   pyBigWig/0.3.18
   rgdal/1.6-6
   scikit-bio/0.5.7
   scikit-learn/1.1.2
   screen_assembly/1.2.7
   sextractor/2.25.0
   shovill/1.1.0
   slow5tools/0.4.0
   snakemake/7.22.0
   snippy/4.6.0
   statsmodels/0.13.1
   tensorboard/2.10.0
   torchvision/0.13.1
   vbz_compression/1.0.2
   vcflib/1.0.3
   versioneer/0.28-Python-3.10.4
   xtb/6.5.1

-------- /apps/easybuild-2022/easybuild/modules/all/Compiler/GCC/11.3.0 --------
   ATLAS/3.10.2-LAPACK-3.10.1         Mash/2.3
   BBMap/39.01                        MetaEuk/6
   BCFtools/1.15.1                    OpenBLAS/0.3.20           (L)
   BEDTools/2.30.0                    OpenMPI/4.1.4             (L)
   BLAT/3.7                           PLINK/2.00a3.6
   BLIS/0.9.0                         POV-Ray/3.7.0.10
   BamTools/2.5.2                     Pysam/0.19.1
   Beast/2.7.3                        SAMtools/1.13
   Bio-DB-HTS/3.01                    SAMtools/1.16.1           (D)
   Bio-SearchIO-hmmer/1.7.3           SLiM/4.0.1
   Bismark/0.24.0                     SOCI/4.0.3
   Boost.Python/1.79.0                SPAdes/3.15.5
   Boost/1.79.0                (D)    STAR/2.7.10b
   Bowtie/1.3.1                       Salmon/1.9.0
   Bowtie2/2.4.5                      SeqLib/1.2.0
   CAT/5.2.3                          Subread/2.0.4
   CD-HIT/4.8.1                       TRUST4/1.0.7
   Clustal-Omega/1.2.4                TransDecoder/5.5.0
   DBD-mysql/4.050                    VCFtools/0.1.16
   DIAMOND/2.1.0                      VEP/107
   FFTW/3.3.10                 (L)    Velvet/1.2.10-mt-kmer_191
   FastANI/1.33                       bcl2fastq2/2.20.0
   FlexiBLAS/3.2.0             (L)    beagle-lib/4.0.0
   GEOS/3.10.3                        bioawk/1.0
   GMAP-GSNAP/2023-02-17              bwa-mem2/2.2.1            (D)
   GSL/2.7                            coverm/0.6.1-linux-x86_64
   GST-plugins-bad/1.20.2             fastp/0.23.2
   GST-plugins-base/1.20.2            gawk/5.1.1
   GStreamer/1.20.2                   kineto/0.4.0
   GTK4/4.7.0                         libxc/5.2.3
   GffCompare/0.12.6                  libxml++/5.0.3
   HTSlib/1.15.1                      lpsolve/5.5.2.11
   IDBA-UD/1.1.3                      pocl/1.8
   Jellyfish/2.3.0                    poppler/22.12.0           (D)
   KMC/3.2.1                          seqtk/1.3
   LAPACK/3.10.1                      sickle/1.33
   Lighter/1.1.2                      tabixpp/1.1.0
   MAFFT/7.505-with-extensions        texlive/20230313          (D)
   MPICH/4.1.2                        vt/0.57721
   MariaDB/10.9.3                     wgsim/20111017

------ /apps/easybuild-2022/easybuild/modules/all/Compiler/GCCcore/11.3.0 ------
   ACTC/1.1
   ANTLR/2.7.7-Java-11
   APR-util/1.6.1
   APR/1.7.0
   ATK/2.38.0
   Apptainer/1.1.8
   Archive-Zip/1.68
   Autoconf/2.71                      (D)
   Automake/1.16.4
   Automake/1.16.5                    (D)
   Autotools/20220317                 (D)
   BWA/0.7.17
   Bazel/5.1.1
   BeautifulSoup/4.10.0
   BioPerl/1.7.8
   Bison/3.8.2                        (D)
   Blosc/1.21.3
   Boost/1.79.0
   Bracken/2.7
   Brotli/1.0.9
   CFITSIO/4.2.0
   CLISP/2.49
   CMake/3.23.1
   CMake/3.24.3                       (D)
   CPLEX/22.1.1
   CapnProto/0.10.2
   CharLS/2.4.1
   Check/0.15.2
   Circos/0.69-9
   Clang/12.0.1
   Clang/13.0.1                       (D)
   Compress-Raw-Zlib/2.202
   ConnectomeWorkbench/1.5.0
   Cython/0.27.3
   Cython/0.29.33                     (D)
   DB/18.1.40
   DB_File/1.858
   DBus/1.14.0
   DMTCP/3.0.0
   DendroPy/4.5.2
   Doxygen/1.9.4
   Eigen/3.3.9
   Eigen/3.4.0                        (D)
   Emacs/28.1
   FASTX-Toolkit/0.0.14
   FFmpeg/4.4.2
   FFmpeg/5.0.1                       (D)
   FLAC/1.3.4
   FLTK/1.3.8
   Flask/2.2.2
   FragGeneScan/1.31
   FreeImage/3.18.0
   FriBidi/1.0.12
   GATK/4.3.0.0-Java-11
   GD/2.75
   GDB/12.1
   GDRCopy/2.3
   GL2PS/1.4.2
   GLM/0.9.9.8
   GLPK/5.0
   GLib/2.72.1
   GLibmm/2.76.0
   GMP/6.2.1
   GObject-Introspection/1.72.0
   GTK2/2.24.33
   GTK3/3.24.33
   GTS/0.7.6
   Gdk-Pixbuf/2.42.8
   Ghostscript/9.56.1
   GitPython/3.1.27
   GnuCOBOL/3.2
   Graphene/1.10.8
   Graphviz/2.50.0
   Graphviz/5.0.0                     (D)
   HDF/4.2.15
   HEALPix/3.82
   HarfBuzz/4.2.1
   ICU/71.1
   IPython/8.5.0
   ISA-L/2.30.0
   ISL/0.24
   ImageMagick/7.1.0-37
   Imath/3.1.5
   JasPer/2.0.33
   JsonCpp/1.9.5
   Judy/1.0.5
   JupyterLab/3.5.0
   LAME/3.100
   LANDIS-II/7.0
   LIBSVM/3.30
   LLVM/14.0.3
   LMDB/0.9.29
   LSD2/2.3
   LZO/2.10
   LibLZF/3.6
   LibTIFF/4.3.0
   LittleCMS/2.13.1
   Lua/5.4.4
   M4/1.4.19                          (D)
   MEGAHIT/1.2.9
   METIS/5.1.0
   MPC/1.2.1
   MPFR/4.1.0
   MUSCLE/5.1.0
   Mako/1.2.0
   Maxima/5.47.0
   Mercurial/6.2
   Mesa/22.0.3
   Meson/0.62.1
   Meson/0.64.0                       (D)
   Mono/6.12.0.122
   NASM/2.15.05
   NCCL/2.12.12-CUDA-11.7.0
   NCCL/2.18.3-CUDA-12.2.0            (D)
   NGS/2.11.2
   NIfTI/2.0.0
   NLopt/2.7.1
   NSPR/4.34
   NSS/3.79
   Nim/1.6.6
   Ninja/1.10.2
   OpenEXR/3.1.5
   OpenJPEG/2.5.0
   OpenPGM/5.2.122
   OpenSlide/3.4.1-largefiles
   OpenVDB/10.0.1
   PAML/4.10.5
   PCRE/8.45
   PCRE2/10.40
   PROJ/9.0.0
   Pango/1.50.7
   Perl/5.34.1-minimal
   Perl/5.34.1                        (D)
   Pillow-SIMD/9.2.0
   Pillow/9.1.1
   PolSpice/3.7.5
   PostgreSQL/14.4
   PyBioLib/1.1.988
   PyCairo/1.21.0
   PyGObject/3.42.1
   PyQt5/5.15.5
   PyYAML/6.0
   Python/2.7.18-bare
   Python/3.10.4-bare
   Python/3.10.4                      (D)
   Qhull/2020.2
   Qt5/5.15.2
   Qt5/5.15.5                         (D)
   Qwt/6.2.0
   RE2/2022-06-01
   RIblast/1.2.0
   RapidJSON/1.1.0
   Rust/1.60.0
   Rust/1.65.0                        (D)
   SCons/4.4.0
   SDL2/2.0.22
   SQLite/3.38.3
   SQLite/3.42.0                      (D)
   SSW/1.1
   SWIG/4.0.2
   Serf/1.3.9
   Subversion/1.14.1
   Subversion/1.14.2                  (D)
   Szip/2.1.1
   TCLAP/1.2.5
   TRF/4.09.1
   Tcl/8.6.12
   Tcl/8.6.13                         (D)
   Tk/8.6.12
   Tkinter/3.10.4
   Trim_Galore/0.6.10
   UCC/1.0.0                          (L)
   UCX-CUDA/1.13.1-CUDA-11.7.0
   UCX-CUDA/1.13.1-CUDA-12.2.0        (D)
   UCX/1.13.1                         (L)
   UDUNITS/2.2.28
   UnZip/6.0
   Voro++/0.4.6
   Wayland/1.20.0
   X11/20220504
   XML-Compile/1.63
   XML-LibXML/2.0207
   XZ/5.2.5-test
   XZ/5.2.5                           (L)
   XZ/5.4.2                           (D)
   XlsxWriter/3.0.8
   Xvfb/1.20.13
   Xvfb/21.1.3                        (D)
   Yasm/1.3.0
   Z3/4.10.2
   ZeroMQ/4.3.4
   Zip/3.0
   any2fasta/0.4.2
   archspec/0.1.4
   argtable/2.13
   assimp/5.2.5
   at-spi2-atk/2.38.0
   at-spi2-core/2.48.3
   binutils/2.38                      (L,D)
   bzip2/1.0.8
   cURL/7.83.0
   cairo/1.17.4
   cppy/1.2.1
   cutadapt/4.2
   dashing/1.0
   datamash/1.8
   dcm2niix/1.0.20220720
   dill/0.3.6
   double-conversion/3.2.0
   elfutils/0.187
   expat/2.4.8
   expecttest/0.1.3
   fastahack/1.0.0
   fastq-pair/1.0
   fermi-lite/20190320
   file/5.43
   filevercmp/20191210
   flatbuffers/2.0.7
   flex/2.6.4                         (D)
   fontconfig/2.14.0
   freeglut/3.2.2
   freetype/2.12.1
   fsom/20141119
   g2clib/1.7.0
   g2lib/3.2.0
   gettext/0.21
   gffread/0.12.7
   giflib/5.2.1
   git/2.36.0-nodocs
   glew/2.1.0
   gnuplot/5.4.4
   googletest/1.11.0
   gperf/3.1
   graphite2/1.3.14
   groff/1.22.4
   gzip/1.12
   help2man/1.49.2
   hifiasm/0.19.5
   hwloc/2.7.1                        (L)
   hypothesis/6.46.7
   intervaltree/0.1
   intltool/0.51.0
   jbigkit/2.1
   jemalloc/5.2.1
   jemalloc/5.3.0                     (D)
   jq/1.5
   jupyter-contrib-nbextensions/0.7.0
   jupyter-server/1.21.0
   kim-api/2.3.0
   libGLU/9.0.2
   libQGLViewer/2.6.4
   libRmath/4.2.1
   libaec/1.0.6
   libaio/0.3.112
   libarchive/3.6.1
   libcerf/2.1
   libdap/3.20.11
   libdeflate/1.10
   libdrm/2.4.110
   libepoxy/1.5.10
   libevent/2.1.12                    (L,D)
   libexif/0.6.24
   libfabric/1.15.1                   (L)
   libffcall/2.4
   libffi/3.4.2
   libffi/3.4.4                       (D)
   libgd/2.3.3
   libgdiplus/6.1
   libgeotiff/1.7.1
   libgit2/1.4.3
   libglvnd/1.4.0
   libgtextutils/0.7
   libharu/2.3.0
   libiconv/1.17
   libidn2/2.3.2
   libjpeg-turbo/2.1.3
   libogg/1.3.5
   libopus/1.3.1
   libpciaccess/0.16                  (L,D)
   libpng/1.6.37                      (D)
   libreadline/8.1.2
   libreadline/8.2                    (D)
   libsigc++/3.4.0
   libsigsegv/2.14
   libsndfile/1.1.0
   libsodium/1.0.18
   libtirpc/1.3.2
   libtool/2.4.7                      (D)
   libunwind/1.6.2
   libvorbis/1.3.7
   libwebp/1.2.4
   libxml2/2.9.10
   libxml2/2.9.13                     (L)
   libxslt/1.1.34
   libyaml/0.2.5
   lxml/4.9.1
   lz4/1.9.3
   make/4.3
   makedepend/1.0.7
   makeinfo/6.7
   makeinfo/6.8                       (D)
   minimap2/2.24
   mm-common/1.0.5
   motif/2.3.8
   multichoose/1.0.3
   ncurses/6.3
   ncurses/6.4                        (D)
   nettle/3.8
   nlohmann_json/3.10.5
   nodejs/16.15.1
   nsync/1.25.0
   numactl/2.0.14                     (L)
   p7zip/17.04
   parallel/20220722
   pigz/2.7
   pixman/0.40.0
   pkg-config/0.29.2
   pkgconf/1.8.0
   pkgconf/1.9.5                      (D)
   pkgconfig/1.5.5-python
   plotly.py/5.12.0
   poppler/22.12.0
   prodigal/2.6.3
   protobuf-python/3.19.4
   protobuf/3.19.4
   pybind11/2.9.2
   python-isal/1.1.0
   qrupdate/1.1.2
   re2c/2.2
   samclip/0.4.0
   scikit-build/0.15.0
   setuptools/63.4.0-Python-3.10.4
   smithwaterman/20160702
   snappy/1.1.9
   snp-sites/2.5.1
   snpEff/5.0e-Java-11
   sparsehash/2.0.4
   tabix/0.2.6
   tbb/2021.5.0
   tcsh/6.24.01
   texinfo/7.0.2
   texlive/20230313
   tqdm/4.64.0
   trimAl/1.4.1
   utf8proc/2.7.0
   util-linux/2.38
   wget/1.21.3
   x264/20220620
   x265/3.5
   xorg-macros/1.19.3                 (D)
   xprop/1.2.5
   xproto/7.0.31
   xxd/8.2.4220
   yaml-cpp/0.7.0
   zlib/1.2.12                        (L)
   zlib/1.2.13                        (D)
   zstd/1.5.2

------------------- /apps/easybuild-2022/Modules/modulefiles -------------------
   mediaflux-data-mover/current (D)    spartan/rhel9              (L,D)
   mediaflux-explorer/current   (D)    spartan/2016-03-parallel
   showq/0.15                          unimelb-mf-clients/current
   slurm/latest                 (L)

--------------- /apps/easybuild-2022/easybuild/modules/all/Core ----------------
   ABAQUS/2022-hotfix-2223                   SHAPEIT/2.r904.glibcv2.17
   ADMIXTURE/1.3.0                           SeqKit/2.3.1
   ANSYS/2023R1                              Trimmomatic/0.39-Java-11
   Anaconda3/2022.10                         VarScan/2.4.4-Java-11
   Autoconf/2.71                             XZ/5.2.5
   Automake/1.16.5                           ant/1.10.11-Java-11
   Autotools/20220317                        ant/1.10.12-Java-11          (D)
   BLAST/2.14.0-Linux_x86_64                 binutils/2.38
   Bison/3.8.2                               bwa-mem2/2.2.1
   CMake/3.12.1                              code-server/4.9.1
   COMSOL/6.0                                cuDNN/8.4.1.50-CUDA-11.7.0
   CUDA/11.5.2                               cuDNN/8.9.3.28-CUDA-12.2.0   (D)
   #!/usr/bin/env bash
module load foss/2022a
module load GCC/11.3.0
module load CUDA/11.7.0
module load OpenMPI/4.1.4
module load Python/3.10.4
module load TensorFlow/2.11.0-CUDA-11.7.0

# check it exist! mkdir /data/gpfs/projects/punim2039/envs
virtualenv /data/gpfs/projects/punim2039/envs/nlp_gpu
source /data/gpfs/projects/punim2039/envs/nlp_gpu/bin/activate

pip install --upgrade pip==23.1.2
pip install six==1.15.0
pip install 'requests>=2.21.0,<3'
pip install pandas==1.2.0
pip install didipack==4.1.2
pip install beautifulsoup4==4.12.2
pip install transformers==4.30.2
pip install html2text==2020.1.16
pip install 'urllib3<2.0'
pip install wrds==3.1.6
pip install nltk==3.8.1
pip install xlrd
pip install openpyxl
pip install pandarallel==1.6.5



                               dotNET-SDK/6.0.101-linux-x64
   CUDA/12.2.0                     (D)       fast5/0.6.5
   CellRanger/7.0.0                          fastGEAR/20161216
   Cereal/1.3.0                              ffnvcodec/11.1.5.2
   EasyBuild/4.7.1                           flex/2.6.4
   EasyBuild/4.8.0                 (D)       foss/2022a                   (L)
   FSL/6.0.6.4                               gettext/0.21
   FastQC/0.11.9-Java-11                     gettext/0.21.1               (D)
   FreeSurfer/7.3.2-centos8_x86_64           gfbf/2022a
   GCC/11.3.0                      (L)       gompi/2022a
   GCCcore/11.3.0                  (L)       gubbins/2.4.0
   Go/1.17.6                                 hwloc/2.7.1
   Go/1.18.3                       (D)       hwloc/2.8.0                  (D)
   IMPUTE2/2.3.2_x86_64_dynamic              imkl/2022.1.0
   IMPUTE2/2.3.2_x86_64_static     (D)       intel-compilers/2022.1.0
   IntelPython/3.9.16-2023.1.0               iomkl/2022a
   Java/8.372                                iompi/2022a
   Java/11.0.18                    (11)      joe/4.6
   Java/17.0.6                     (D:17)    libevent/2.1.12
   Julia/1.8.5-linux-x86_64                  libpciaccess/0.16
   M4/1.4.19                                 libpng/1.6.37
   MATLAB/2023a_Update_1                     libtool/2.4.7
   MCR/R2016a                                libxml2/2.9.13
   META/1.7                                  libxml2/2.10.3               (D)
   Mambaforge/23.1.0-4                       manta/1.6.0
   Mathematica/13.2.1                        mediaflux-data-mover/current
   Maven/3.9.3                               mediaflux-explorer/current
   Miniconda2/4.7.10                         ncurses/6.2
   Miniconda3/22.11.1-1                      ncurses/6.3
   MosiacHunter/1.0.0                        numactl/2.0.14
   NVHPC/22.11-CUDA-11.7.0                   numactl/2.0.16               (D)
   Nextflow/23.04.2                          picard/2.25.1-Java-11
   OpenSSL/1.1                     (L)       picard/3.0.0-Java-17         (D)
   OptiX/7.6.0                               pkgconf/1.8.0
   PMIx/3.2.3                                snptest/2.5.6
   PMIx/4.2.2                      (L,D)     tbl2asn/20220427-linux64
   Pandoc/3.1.2                              xorg-macros/1.19.3
   Pilon/1.23-Java-11                        zlib/1.2.12
   QIIME2/2022.11

  Where:
   L:        Module is loaded
   Aliases:  Aliases exist: foo/1.2.3 (1.2) means that "module load foo/1.2" will load foo/1.2.3
   D:        Default Module

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching
any of the "keys".


