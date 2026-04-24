# A User-Friendly Tool to Automate Assessment of Housing Recovery Following a Tornado

![Frame Work](assets/Framework.png)
**Figure 2**: Automated Recovery Assessment Framework. 

## Step 1: Gather Input Data
* Spatial Videos
* GIS files
* Parcel Polygons
* House Polygons

## Step 2: Process Data
* First, frames will be extracted from spatial videos at a 1-second interval
* Process Spatial Data Toolbox will be used to:
  * Match the frame with the GIS
  * Transform and map the GIS to the corresponding parcels
  * Group frames into Address Folders (i.e., frames containing scene of the house)

## Step 3: Label Data

**Table 1**. Criteria for Labeling Frames for the Classification Model
| Recovery States | Elaboration |
|:---|:---|
| *1-Uninhabited* | Debris and collapse, and moderate damage to houses can be seen. |
| *2-Empty* | Debris has been cleaned up; grass has grown. However, no sign of reconstruction can be seen. |
| *3-Rebuilding* | Lots are being cleared to construct foundations; wall enclosures are up, and housewrap can be seen. |
| *4-Rebuilt* | Good as new. Slight or minor damage to nonstructural components. |

![Samples](assets/samples.png)
**Figure 3**: Samples of Training Datasets: (a) Uninhabited; (b) Empty; (c) Rebuilding; (d) Rebuilt. 

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




