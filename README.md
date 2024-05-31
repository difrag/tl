# [Check your data](https://check-ur-data.streamlit.app/) - a steamlit webapp
## Description
A web app created for a university project that enables you to upload tabular datasets and get insights related to data statistics, missing values and data distribution. Machine learning training based on the uploaded data is also available to the user. Latest update also allows comparing results between different machine learning methods.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Creators](#creators)

## Installation
Step-by-step instructions on how to set up the project locally.

### Prerequisites
Make sure you have the following software installed if you are running on windows (if not you already know what to do):
- **Git**: [Download Git](https://git-scm.com/)
- **Docker**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
- **WSL2**: [Download WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

### Setup
1. **Clone the Repository**
Open a terminal and run the following command:
```bash
git clone https://github.com/difrag/tl
cd tl
```
2. **Build the Docker image**
```bash
docker-compose build
```

3. **Run the Docker container**
```bash
docker-compose up
```
## Usage
Open the streamlit [link](https://check-your-data.streamlit.app/) or follow the [Installation](#installation) steps for localhost usage and upload the dataset of your choice. Datasets used for testing during development : [Thyroid_Diff.csv](https://github.com/difrag/tl/files/15456731/Thyroid_Diff.csv) and [iris.csv](https://github.com/difrag/tl/files/15456730/iris.csv) You could use one of these to get started if you do not have one already. After the dataset is uploaded you are given a preview of the data uploaded and some options to configure the machine learning models. Different tabs are created to make the app navigation more intuitive.\

## Creators
- **Fragkoulis Dimitris** - Π2015191
- **Nikolopoulos Konstantinos** - Π2016051
- **Christos Grigorakos** - Π2020146


