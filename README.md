<h1 align="left"> TornadoNET: A Deep Learning Approach for Geospatial Data to Automate Housing Recovery Assessment Using the 2011 Joplin Tornado and 2021 Mayfield Tornado Datasets </h2>

<h4 align="left">by <a href="">Tuan Tran</a>, <a href="https://scholar.google.com/citations?user=HxlMZr8AAAAJ&hl=en">Abdullah Braik</a>, <a href="https://scholar.google.com/citations?user=VsgAGKQAAAAJ&hl=en">Maria Koliou</h4>


![Frame Work](assets/Framework.png)
**Figure 1**: Automated Recovery Assessment Framework. 

## Step 1: Gather Input Data
* Spatial Videos
* GIS files
* Parcel Polygons
* House Polygons

## Step 2: Process Data
* Extract frame at an interval of choices (e.g. 1 second)
* [Process Spatial Video & GIS Toolbox](toolbox/Process_Spatial_Video_&_GIS.pyt) will be used to:
  * Match the frame with the GIS
  * Group frames into Address Folders (i.e., frames containing views of the house)

## Step 3: Label Data
| Recovery States | Elaboration |
|:---|:---|
| *1-Uninhabited* | Debris and collapse, and moderate damage to houses can be seen. |
| *2-Empty* | No sign of reconstruction can be seen. |
| *3-Rebuilding* | Foundation construction, wall enclosures, and housewrap can be seen. |
| *4-Rebuilt* | Good as new. Slight or minor damage to nonstructural components. |

![Samples](assets/samples.png)
**Figure 2**: Samples of Training Datasets: (a) Uninhabited; (b) Empty; (c) Rebuilding; (d) Rebuilt. 

## Step 4: Split Train/Validation Dataset

## Step 5: Train Deep Learning Models

## Step 6: Classify the Data

## Step 7: Analyze Longitudinal Results

## Installation
1. Clone this repository

2. Install dependencies

3. Run setup from the repository root directory

4. 

## Citation
Use this bibtex to cite this repository




