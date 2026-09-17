# UK Node of the Digital Earths Global Hackathon

A group of experiments have been conducted using the Met Office Unified Model (MetUM) with a focus on the DYAMOND-3 period (Jan 2020-Feb 2021). While these experiments include standalone explicit convection global simulations we have also developed a cyclic tropical channel and include limited area model simulations to build our understanding of how resolving smaller-scale processes feeds back on to the large-scale atmospheric circulation.

## MetUM Experiments
 1) N2560 Global simulation using RAL3p3 (Regional Atmostphere-Land v3.3) physics configuration (explicit convection). This simulation also has a +4K SST variant.
 2) N1280 Global simulation using GAL9 (Global Atmosphere-Land v9) physics configuration (parameterised convection). This experiment also has nested cyclic tropical channel (CTC) and limited area model simulations (LAMs) (all 4.4 km resolution) using the RAL3p3 physics configuration. 
 3) N1280 and N2560 Global simulations using CoMA9_TBv1p2 (CoMorphA9 trailblazer version 1.2 configuration) - a new convection parameterisation that has been developed in the MetUM and tuned for running at ~5 km resolution globally. The N2560 simulation also has a +4K SST variant. 

## Regions (brackets show shortnames used in filenames)
 - Global (glm)
 - Tropical (CTC) - nested inside N1280 GAL9 simulation
 - South America (SAmer) - nested inside N1280 GAL9 simulation
 - Africa (Africa) - nested inside N1280 GAL9 simulation
 - South East Asia (SEA) - nested inside N1280 GAL9 simulation

## Science/Model
### GAL9
Global atmosphere-land version 9 is the latest parameterised convection configuration of the MetUM and will soon be used for Met Office operational global NWP forecasts. This science configuration parameterises convection and has been thoroughly tested for global simulations at both NWP and climate length simulations i.e. it is the most mature MetUM science configuration for use in global simulations. We use this science configuration to run an N1280 Global simulation. The latest available model description paper for GAL science configuration is [Walters et al., 2019 (GA7)] (https://gmd.copernicus.org/articles/12/1909/2019/).

### RAL3p3
Regional atmosphere-land version 3.3 is the latest explicit convection configuration of the MetUM. RAL3p3 includes the double-moment CASIM microphysics scheme and the bimodal large-scale cloud scheme, notably there are no changes to science settings between tropical and mid-latitude regions so RAL3 can be used to experiment with explicit convection global simulations. For a full description of the science settings used in RAL3p3 simulations see [Bush et al., 2024](https://gmd.copernicus.org/preprints/gmd-2024-201/). This science configuration is used in the CTC and limited area model experiments and a N2560 (~5 km) global simulation. A +4K SST variant is also available for the same period.

### CoMA9_TBv1p2
CoMorph-A is a new convection scheme that has been developed and tested within the Met Office Unified Model [e.g. Lock et al., 2024](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.4781). CoMorph is a mass-flux scheme and allows convection to initiate from all levels, it includes the ability for multiple plumes to coexist in a grid box. CoMorph-A9_TBv1p2 has been tuned for running at approximately 5 km horizontal resolution globally. N1280 and N2560 (~5 km) Global simulations are available, and a +4K SST variant of the N2560 simulation is also available for the same period. This science configuration is built on top of GAL9 but using the new CoMorphA convection scheme - it has not yet been as thorougly tested at the Met Office as (e.g.) the GAL9 scheme. 
