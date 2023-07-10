# Persons unsupervised recognition via clusterization

## 1. Create encodings

Create encodings of faces in images. If there are no faces in image then save empty encoding vector. If several faces are found then only the first one will be selected.

```
python calc_encodings.py --dataset=data/clusters --encodings=encodings.json
```

## 2. Run clusterization

Run demo.ipynb, V-score=0.94.

![cluster rows](image/README/1688918547009.png)
![cluster map](image/README/1688918564991.png)
