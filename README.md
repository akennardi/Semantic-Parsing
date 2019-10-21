# Repository for COMP8755 Individual Computing Project

**Project Title**

Domain Adaptation for Low-Resource Neural Semantic Parsing

**Supervisor**

Dr. Qing Wang and Dr. Gabriela Ferraro

**Instruction**

**Folder**

1. Folder `Data` consists of all data sets used in the experiment

2. Folder `Evaluation` consists of all scripts to run evaluation and plot.
After running the experiment, the log file in `.txt` format can be found in Sub-folder `Output`
After running plot script, the plot in `pdf` can be found in Sub-folder `Plot`

3. Folder `Experiment` consists of all shell scripts to run the experiment

4. Folder `Preprocess` consists of all scripts to perform pre-processing of data set

5. Folder `Src` consists of source code to train the model via transfer learning and without transfer learning
 
**Experiments**

All the pre-processing step and experiments are stored in `Experiment` Folder.
Therefore, the code should be run from this folder

1. To train model without transfer learning, run following script from terminal/ command line

    * `sh Train_Subset_GEO.sh` for anonymised GeoQuery
    * `sh Train_Subset_GEO_EXP.sh` for un-anonymised GeoQuery
    * `sh Train_Subset_GEO_EXP_Query.sh` for un-anonymised GeoQuery with Query-Split
    
2. To train model via transfer learning, run following script from terminal/ command line
    * `sh Transfer_learning_ATIS_to_GEO.sh` for anonymised GeoQuery
    * `sh Transfer_learning_ATIS_to_Geo_EXP.sh` for un-anonymised GeoQuery
    * `sh Transfer_learning_ATIS_to_Geo_EXP_QUERY.sh` for un-anonymised GeoQuery with Query-Split

3. To generate plot in the report, run following script
    * `sh Plot_All.sh`

**Acknowledgement**

The code implemented here are based on the implementation from [here](https://github.com/Alex-Fabbri/lang2logic-PyTorch).

**Software and Package Requirement**: 

See the `requirements.txt`