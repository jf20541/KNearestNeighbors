# KNearestNeighbors

## Objective
Optimize two hyperparameters (K-values and Distance Function) for K-Nearest Neighbor Model.

Distance Function: [Euclidean, Minkowski, Manhattan, Hamming]\
K-Value: output has 3 classifications, recommend an even K-Value

## Model
K-Nearest Neighbors: A non-parametric classification model that calculates the distance of n-test observations from all the observations of the training dataset and output as the class with the highest frequency from the K-most similar instances.

KK Disciplines\
Lazy Learning: Training is not required and all of the work happens at the time a prediction is requested.\
Instance-Based Learning: Raw training instances are used to make predictions.\
Non-Parametric: KNN makes no assumptions about the functional form of the problems being solved.

Avoid\
Curse of Dimensionality: As the number of dimensions increases the volume of the input space increases at an exponential rate

## Repository File Structure

    ├── src          
    │   └── main.py              # Optimize two hyperparameters K-values with variants of Distance Function for K-Nearest Neighbor Model
    ├── plots
    │   └── ErrorRatekValue.png  # Error Rate K-Value
    ├── requierments.txt         # Packages used for project
    └── README.md

## Outputs & Distance Functions

- Euclidean Distance\
![](https://latex.codecogs.com/gif.latex?d%28x%2C%20y%29%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_%7Bi%7D-y_%7Bi%7D%29%5E%7B2%7D%7D)
```bash
Euclidean Distance
K-Nearest Neighbor Accuracy: 96.67%
```
- Minkowski Distance\
![](https://latex.codecogs.com/gif.latex?d%28x%2C%20y%29%3D%20%5Cleft%20%28%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%7Cx_%7Bi%7D-y_%7Bi%7D%20%5Cright%20%7C%5E%7Bp%7D%5Cright%29%5E%7B1/p%7D)
```bash
Minkowski Distance
K-Nearest Neighbor Accuracy: 96.67%
```
- Manhattan Distance\
![](https://latex.codecogs.com/gif.latex?d%28x%2Cy%29%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%20%7C%20x_%7Bi%7D-%20y_%7Bi%7D%20%5Cright%20%7C)
```bash
Manhattan Distance
K-Nearest Neighbor Accuracy: 93.33%
```
```bash
Hamming Distance
K-Nearest Neighbor Accuracy: 93.33%
```

## Metric
![](https://latex.codecogs.com/gif.latex?Accuracy%3D%5Cfrac%7Btp&plus;tn%7D%7B%28tp%20&plus;%20tn%29&plus;%28fp-fn%29%29%7D)\
tp = True Positive\
tn = True Negative\
fp = False Positive\
fn = False Negative

## Data
```bash
Target Class

Iris-setosa:      float64
Iris-versicolor:  float64
Iris-virginica:   float64
```
```bash
Features     (normalized)

Sepal-width:      float64
Sepal-length:     float64
Petal-width:      float64
Petal-length:     float64
```
## Sources
https://www.kdnuggets.com/2020/11/most-popular-distance-metrics-knn.html
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978658/#:~:text=K%2Dnearest%20neighbor%20(k%2D,decide%20the%20final%20classification%20output.
