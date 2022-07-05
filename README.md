# A product-component framework for modelling stock dynamics and its application for electric vehicles batteries and lithium-ion
## Documentation of the code used for the different modelling approaches

This repository contains the data and algorithms used to compute the different modelling approaches presented in the related publication (add link here if ever published).

The main files contaned here are:
1. product_component_model.py: Python class for handling of stock dynamics of product-component-systems. This is the main script containing the generic definitions for each modelling approach. It can be used to compute the stock dynamics of a product-component-system under any of the 12 modelling approaces introduced.
2. case_study.ipynb: Notebook containing the case study presented in the manuscript and supplementary information.
3. generic_cohort_model.ipynb: Notebook presenting the different modelling approaches for a generic stock that highlights the differences in outcomes for a stock that initially increases, then stabilizes, and ultimately decreases again.
4. cases_visualizations.ipynb: Notebook containing a multi visualization of all cases under the same lifetime assumptions but where the dynamics where calculated using the different modelling options for a generic stock as presented in point 3.

In addition, the data folder contains all inout data used for this study and relevant output data.
