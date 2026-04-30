<h1 align="left"> TornadoNET: A Deep Learning Approach for Geospatial Data to Automate Housing Recovery Assessment Using the 2011 Joplin Tornado and 2021 Mayfield Tornado Datasets </h2>

<h4 align="left">by <a href="">Tuan Tran</a>, <a href="https://scholar.google.com/citations?user=HxlMZr8AAAAJ&hl=en">Abdullah Braik</a>, <a href="https://scholar.google.com/citations?user=VsgAGKQAAAAJ&hl=en">Maria Koliou</h4>


![Frame Work](assets/Framework.png)
**Figure 1**: Automated Recovery Assessment Framework. 

## Step 1: Gather Input Data
* Spatial Videos
* GIS files 
* Parcel Polygons
* House Polygons

Demo data can be found [here](demo)

## Step 2: Process Data
* Extract frame using the [extract_frame.ipynb](toolbox/extract_frames.ipynb) toolbox.
* [Process Spatial Video & GIS Toolbox](toolbox/Process_Spatial_Video_&_GIS.pyt) will be used to:
  * Match the frame with the GIS
  * Group frames into Address Folders (i.e., frames containing views of the house)
 
![toolbox_process_spatial](assets/toolbox_process_spatial.png)

## Step 3: Label Data
| Recovery States | Elaboration |
|:---|:---|
| *1-Uninhabited* | Debris and collapse, and moderate damage to houses can be seen. |
| *2-Empty* | No sign of reconstruction can be seen. |
| *3-Rebuilding* | Foundation construction, wall enclosures, and housewrap can be seen. |
| *4-Rebuilt* | Good as new. Slight or minor damage to nonstructural components. |

![Samples](assets/samples.png)
**Figure 2**: Samples of Training Datasets: (a) Uninhabited; (b) Empty; (c) Rebuilding; (d) Rebuilt. 

## Step 4: Split Datasets
Once all the data are labeled, you can split the datasets into training, validation, and test sets geographically in ArcGIS Pro using the [Split_Datasets.pyt](toolbox/Split_Datasets.pyt) toolbox.

![toolbox_split_datasets](assets/toolbox_split_datasets.png)
## Step 5: Train Deep Learning Models
* I have set up 4 different models in [deep_learning](deep_learning). Make sure you download the entire folder to use the notebooks:
** [ResNet-50](deep_learning/resnet50.ipynb)
** [Convolutional Neural Network-Long Short-Term Memory](deep_learning/cnn_lstm.ipynb)
** [Convolutional Neural Network-Long Short-Term Memory](deep_learning/cnn_stm.ipynb)
** [Swin Transformer V2-Base](deep_learning/swin.ipynb)

Additionally, if you want to explore Grad-CAM analysis on data, please refer to [gradcam](deep_learning/grad_cam)

## Step 6: Classify the Data
Once you train the deep learning models, use the .PTH file to perform large-scale mapping of the recovery dataset.

![toolbox_perform_recovery_assessment](assets/toolbox_perform_recovery_assessment.png)

## Step 7: Analyze Longitudinal Results
After you perform large-scale mappings of the recovery dataset over multiple time periods, you can perform a longitudinal analysis using the [Perform_Longitudinal_Analysis.pyt](toolbox/Perform_Longitudinal_Analysis.pyt)

## Citation
Use this bibtex to cite this repository




