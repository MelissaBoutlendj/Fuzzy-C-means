# Fuzzy C-means for fuzzy image segmentation

Lab of the course *Advanced Pattern Recognition* (M2 VMI, Université Paris Cité).

Implementation of Fuzzy C-means (NumPy) and application to the segmentation
of a grey-level and a colour image into fuzzy regions.

## Run
```
pip install numpy scikit-image matplotlib
python3 fcm_tp.py --gray milky-way-nvg.jpg --color milky-way.jpg
```
Figures are saved in `results/`.
