Team ID: TM-34

Members:<br>
Chow Wei Cong<br>
Chua Yam Heng<br>
Chu Yi Xin<br>
<br>
In misc/model selection, models.ipynb contains the 7 models given in the workshop as well as a combined graph showing the effectiveness of the graph using Balanced Accuracy, F1 Macro and Prediction Time.<br>
models_2.ipynb contains the 2 shortlisted models and their improved versions as we tried to fine tune both.<br>
The two model_testing.py and model_training.py files are the final model we chose.<br>
To use, run model_training.py and it should generate a new gradient_boosting_ctg_model.pt as well as the top 10 most important features. Then run model_testing.py and it will show you the accuracy and heatmap of the model.
