# CMEPDA_exam

CMEPDA_exam is a repository containing python programs that implement the analysis of raw data taken from acquisitions with a PET- detector. Positron emission tomography (PET) is a functional imaging technique that uses radioactive substances known as radiotracers to visualize and measure changes in metabolic processes. PET measures the two annihilation photons that are produced back-to-back after positron emission by the radiotracer. Scintillation pixelated detectors are used as detection elements. Due to the positron annihilation we expect to observe two photons at roughly the same time (in coincidence) in the detector. The final aim of this project is to obtain a file containing all the coincidence events that were detected by the acquisition passing through some steps. The project is made by two python scripts: 'PET_analyser.py' and 'functionbox.py'. The first one is used to compute the analysis using all the functions contained in 'functionbox.py'.

# How to use it
In a nutshell, you should be able to compute 'PET_analyser.py' using python. The input raw file is a .dat.dec, a data file that have been decodified to be used in the analysis. To understand all the functionalities and options of the program is useful to use the '--help' function in this way:

```
python PET_analyser.py --help
```

obtaining this output on the nutshell:

```
usage: PET_analyser.py [-h] [-f INPUT_FILE] [-c COUNT_EVENTS] [-cw COINCIDENCES_WINDOW] [-p PEDESTALS_FILENAME] [-tdc TDC_CALIBRATION_FILENAME] [-lut CRYSTAL_MAP_FILENAME] [-showmaps SHOW_FLOODMAP_LUT] [-calib ENERGY_CALIBRATION] [-o OUTFILE] [-ris RESOLUTION_ARRAY] [-CTR COINCIDENCE_TIME_RESOLUTION] [-spectrum PLOT_PIXEL_SPECTRA] [-ew ENERGY_WINDOW] [-mp MULTIPROCESSING]

This program takes in input a raw file and gives in output a coincidences file

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input_file INPUT_FILE
                        input file to process
  -c COUNT_EVENTS, --count_events COUNT_EVENTS
                        size of subfile to process
  -cw COINCIDENCES_WINDOW, --coincidences_window COINCIDENCES_WINDOW
                        lenght of coincidences window in ns
  -p PEDESTALS_FILENAME, --pedestals_filename PEDESTALS_FILENAME
                        json file that stores pedestals value
  -tdc TDC_CALIBRATION_FILENAME, --tdc_calibration_filename TDC_CALIBRATION_FILENAME
                        json file that stores tdc calibration values
  -lut CRYSTAL_MAP_FILENAME, --crystal_map_filename CRYSTAL_MAP_FILENAME
                        json file that stores LUT
  -showmaps SHOW_FLOODMAP_LUT, --show_floodmap_LUT SHOW_FLOODMAP_LUT
                        Show floodmaps and LUT for each ASIC
  -calib ENERGY_CALIBRATION, --energy_calibration ENERGY_CALIBRATION
                        json file that stores energy calibration values
  -o OUTFILE, --outfile OUTFILE
                        output file containing coincidences events
  -ris RESOLUTION_ARRAY, --resolution_array RESOLUTION_ARRAY
                        array containing energy resolution values
  -CTR COINCIDENCE_TIME_RESOLUTION, --coincidence_time_resolution COINCIDENCE_TIME_RESOLUTION
                        compute and show histogram of time differences
  -spectrum PLOT_PIXEL_SPECTRA, --plot_pixel_spectra PLOT_PIXEL_SPECTRA
                        this is a list where the first element is True/False and the remains stand for [[TX], [ASIC]]
  -ew ENERGY_WINDOW, --energy_window ENERGY_WINDOW
                        this is a list where the first element is True/False and the remains stand for [energy_min, energy_max]
  -mp MULTIPROCESSING, --multiprocessing MULTIPROCESSING
                        activate multiprocessing to compute "pedestal_and_tdc" function
```

Some results acquired during the analysis are reported as example.

When 'showmaps' is set to TRUE we can obtain an image showing the floodmap and the consequent operation computed on the data to recognize the pixels. This operation is computed using a watershed segmentation algorithm from scikit-image.

<img src="https://github.com/RebeccaAnzalone/CMEPDA_exam/IMAGES_readme/tx13asic11.png" height="250" width="250">

Other important results that can be reached using this project are the energy spectrum, the energy resolution of the system and the Coincidence Time Resolution (CTR) from the spectrum of the coincidences' time differences.

<img src="https://github.com/RebeccaAnzalone/CMEPDA_exam/IMAGES_readme/energy.png" height="250" width="250">

<img src="https://github.com/RebeccaAnzalone/CMEPDA_exam/IMAGES_readme/CTR.png" height="250" width="250">

---

**These modules have been created using Python 3.8.10**

***Please, read the documentation (badge below) and requirements ([requirements](https://github.com/RebeccaAnzalone/CMEPDA_exam/requirements.txt)) for further details.***

---

[![Documentation Status](https://readthedocs.org/projects/pet-data-analysis/badge/?version=latest)](https://pet-data-analysis.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/RebeccaAnzalone/CMEPDA_exam/actions/workflows/unittests.yml/badge.svg)](https://github.com/RebeccaAnzalone/CMEPDA_exam/actions/workflows/tests.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/RebeccaAnzalone/CMEPDA_exam)
