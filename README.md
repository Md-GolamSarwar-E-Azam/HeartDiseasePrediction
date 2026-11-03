##  End to End Heart Disease Prediction App: CardioCheck App

## Run from terminal:

docker build -t mlopsreg.azurecr.io/cardiocheck:latest .

docker login mlopsreg.azurecr.io

docker push mlopsreg.azurecr.io/cardiocheck:latest

