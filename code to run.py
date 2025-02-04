# prompt: write code to install pycm
!pip install pingouin
!pip install pycm
import os

# Change directory to the desired path
os.chdir('/content/drive/MyDrive/Colab Notebooks/code_exp')
import numpy as np
import warnings
import train_eval as te
import experiments as exps

compas = {
    'tag': 'UN',
    'name': 'UN_peace_keeping',
    'filename': '/content/drive/MyDrive/Colab Notebooks/code_exp/Data/test - Copy.csv',
    'sensitive_attribute': {'region_Africa','region_Americas','region_Asia and the Pacific','region_Europe','region_Middle East'},
    'target': 'nat_civilian_staff',
    'categorical_attributes': {'Type_complex multidimensional peacekeeping mission','Type_traditional peacekeeping mission', 'location_Abyei','location_Bosnia and Herzegovina','location_Burundi','location_Central African Republic','location_Chad, Central African Republic','location_Cyprus','location_Darfur','location_Democratic Republic of the Congo','location_East Timor','location_Ethiopia, Eritrea','location_Georgia',
                               'location_Golan','location_Haiti','location_India, Pakistan','location_Iraq, Kuwait','location_Israel','location_Ivory Coast',
                               'location_Kosovo','location_Lebanon','location_Liberia','location_Mali',
                               'location_Prevlaka Peninsula','location_Sierra Leone','location_Sudan','location_Syria','location_Western Sahara',
                                'Multinational Forces', 'Use of Force', 'Civilian Component', 'Capacity Building','Protection of Civilians','Collaboration with Local Actors','Monitoring and Reporting','Exit Strategy'},
    'numerical_attributes': ['int_civilian_staff' ],
    'minima_numerical_attributes': {}
}
np.random.seed(0)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
# this ignores warnings for the computation of IRR statistics in the case of zero noise.
attribute=['region_Africa','region_Americas','region_Asia and the Pacific','region_Europe','region_Middle East']
trained_models_folds = te.fit_model_data(compas,attribute)
parameters_experiments_1 = {
    'probabilities': np.arange(0, 0.301, 0.01),
    'grouped': 'Y',
    'num_minima': 'N',
    'feature_set': 'all',
    'variance_list': [1, 5, 10],
    'variance':[]
}

exps.experiments(compas, parameters_experiments_1, trained_models_folds,attribute)

# Change directory to the desired path
