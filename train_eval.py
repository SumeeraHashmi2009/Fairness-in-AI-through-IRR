import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pingouin as pg
from pycm import ConfusionMatrix # see https://www.pycm.io/doc/
import numpy as np
import perturbations as pt
from sklearn.linear_model import LinearRegression

# extract dataset information
def prepare_data(dataset_info,attribute):
    data = pd.read_csv(dataset_info['filename'])
  
    target = dataset_info['target']
    at = dataset_info['sensitive_attribute']
    Y = data[target]

    X = data.drop([target], axis=1)
    A = data[attribute]
    return X, A, Y

# generate 5 folds of data
def get_5fold_data(X, A, Y):
    folds = []
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X)   
    for train_index, test_index in kf.split(X):    
        CV_X = X.iloc[train_index]
        CV_A = A.iloc[train_index]
        CV_Y = Y[train_index]
        
        holdout_X = X.iloc[test_index]
        holdout_A = A.iloc[test_index]
        holdout_Y = Y[test_index]
        
        fold = {'CV_X': CV_X, 'CV_Y': CV_Y,'holdout_X': holdout_X,'holdout_Y': holdout_Y, 'CV_A': CV_A, 'holdout_A': holdout_A}
        folds.append(fold)
    return folds

# fit five logistic regressions to the five folds
def fit_model(folds, dataset_info):
    for fold in folds:
        CV_X = fold['CV_X']
        CV_Y = fold['CV_Y']
        
        scaler = StandardScaler()
        numerical = dataset_info['numerical_attributes']
        CV_X = CV_X.copy()
        CV_X.loc[:, numerical] = scaler.fit_transform(CV_X[numerical])
        model = LinearRegression()
        lr = model.fit(CV_X, CV_Y)
        fold['model'] = lr
        fold['scaler'] = scaler
    return folds

# fit five models directly from data
def fit_model_data(dataset_info,attribute):
    X, A, Y = prepare_data(dataset_info,attribute)
    folds = get_5fold_data(X, A, Y)
    folds = fit_model(folds, dataset_info)
    return folds   

# get predictions and probabilites for the five test folds for one noise level
def get_results(folds, proba, dataset_info, parameter_settings,attribute):
    var = parameter_settings['variance']
    feature_list_num = parameter_settings['features_num']
    feature_list_cat = parameter_settings['features_cat']
    grouped = parameter_settings['grouped']
    num_minima = parameter_settings['num_minima']
    
    #at = dataset_info['sensitive_attribute']
    #attribute= list(at)
    minima_numerical_attributes = dataset_info['minima_numerical_attributes']
    
    outputs_folds = []
    
    for fold in folds:
        #for atr in attribute:
            outputs = {}
            
            holdout_X = fold['holdout_X']
            holdout_A = fold['holdout_A']

            holdout_X = holdout_X.copy()
            holdout_X_perturbed = holdout_X.copy()
            
            pt.perturb_total(holdout_X_perturbed, attribute, feature_list_num, feature_list_cat, var, proba, grouped, num_minima, minima_numerical_attributes)
            
            scaler = fold['scaler']        
            lr = fold['model']
            numerical = dataset_info['numerical_attributes']
            
            holdout_X.loc[:, numerical] = scaler.transform(holdout_X[numerical])
            holdout_X_perturbed.loc[:, numerical] = scaler.transform(holdout_X_perturbed[numerical])
            
            

            predictions = lr.predict(holdout_X)

            predictions_perturbed = lr.predict(holdout_X_perturbed)
            
            outputs['preds'] = predictions

            outputs['preds_p'] = predictions_perturbed

               
            outputs[attribute[0]] = holdout_A['region_Africa'].to_numpy()
            outputs[attribute[1]] = holdout_A['region_Americas'].to_numpy()
            outputs[attribute[2]] = holdout_A['region_Asia and the Pacific'].to_numpy()
            outputs[attribute[3]] = holdout_A['region_Europe'].to_numpy()
            outputs[attribute[4]] = holdout_A['region_Middle East'].to_numpy()
            print (outputs[attribute[0]])
            outputs = pd.DataFrame(outputs)
            outputs_folds.append(outputs)
    
    return outputs_folds

# calculate irr metrics on five folds for one noise level, then compute mean of five folds
def get_metrics(outputs_folds, dataset_info,attribute):
    

    metrics_folds = []

    
    for outputs in outputs_folds:
        df = pd.DataFrame(outputs)
        df.to_csv('/content/drive/MyDrive/Colab Notebooks/code_exp/Data/output_file - Copy.csv', index=False)       
        metrics = {}
        

        out_g0 = outputs[outputs[attribute[0]]==1]
        out_g1 = outputs[outputs[attribute[1]]==1]
        out_g2 = outputs[outputs[attribute[2]]==1]
        out_g3 = outputs[outputs[attribute[3]]==1]
        out_g4 = outputs[outputs[attribute[4]]==1]
        

        

       

        # transform data into long form for ICC computation with pingouin package
        pred_group_0 = out_g0[['preds', 'preds_p']].copy()
        pred_group_1 = out_g1[['preds', 'preds_p']].copy()
        pred_group_2 = out_g2[['preds', 'preds_p']].copy()
        pred_group_3 = out_g3[['preds', 'preds_p']].copy()
        pred_group_4 = out_g4[['preds', 'preds_p']].copy()
        
        pred_group_0['index'] = pred_group_0.index
        pred_group_1['index'] = pred_group_1.index
        pred_group_2['index'] = pred_group_2.index
        pred_group_3['index'] = pred_group_3.index
        pred_group_4['index'] = pred_group_4.index
       
        pred_group_0 = pd.melt(pred_group_0, id_vars=['index'], value_vars=list(pred_group_0)[:-1])
        pred_group_1 = pd.melt(pred_group_1, id_vars=['index'], value_vars=list(pred_group_1)[:-1])
        pred_group_2 = pd.melt(pred_group_2, id_vars=['index'], value_vars=list(pred_group_2)[:-1])
        pred_group_3 = pd.melt(pred_group_3, id_vars=['index'], value_vars=list(pred_group_3)[:-1])
        pred_group_4 = pd.melt(pred_group_4, id_vars=['index'], value_vars=list(pred_group_4)[:-1])
        
        # compute ICC statistics using pingouin (yields table)
        icc_pgroup_0 = pg.intraclass_corr(data=pred_group_0, targets='index', raters='variable', ratings='value')
        icc_pgroup_1 = pg.intraclass_corr(data=pred_group_1, targets='index', raters='variable', ratings='value')
        icc_pgroup_2 = pg.intraclass_corr(data=pred_group_2, targets='index', raters='variable', ratings='value')
        icc_pgroup_3 = pg.intraclass_corr(data=pred_group_3, targets='index', raters='variable', ratings='value')
        icc_pgroup_4 = pg.intraclass_corr(data=pred_group_4, targets='index', raters='variable', ratings='value')
        
        # ICC 2
        metrics['group_0_ICC_3'] = icc_pgroup_0['ICC'][1]
        metrics['group_1_ICC_3'] = icc_pgroup_1['ICC'][1]
        metrics['group_2_ICC_3'] = icc_pgroup_2['ICC'][1]
        metrics['group_3_ICC_3'] = icc_pgroup_3['ICC'][1]
        metrics['group_4_ICC_3'] = icc_pgroup_4['ICC'][1]
        
        #CCC
        y_true=list(out_g0['preds'].copy())
        y_pred=list(out_g0['preds_p'].copy())
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc_pgroup_0=(2 *  covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        metrics['group_0_CCC_1'] = ccc_pgroup_0
        y_true=list(out_g1['preds'].copy())
        y_pred=list(out_g1['preds_p'].copy())
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covariance1 = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc_pgroup_1=(2 *  covariance1) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        y_true=list(out_g2['preds'].copy())
        y_pred=list(out_g2['preds_p'].copy())
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covariance2 = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc_pgroup_2=(2 *  covariance2) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        y_true=list(out_g3['preds'].copy())
        y_pred=list(out_g3['preds_p'].copy())
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covariance3 = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc_pgroup_3=(2 *  covariance3) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        y_true=list(out_g4['preds'].copy())
        y_pred=list(out_g4['preds_p'].copy())
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covarinace4 = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc_pgroup_4=(2 *  covarinace4) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        metrics['group_0_CCC_1'] = ccc_pgroup_0
        metrics['group_1_CCC_1'] = ccc_pgroup_1
        metrics['group_2_CCC_1'] = ccc_pgroup_2
        metrics['group_3_CCC_1'] = ccc_pgroup_3
        metrics['group_4_CCC_1'] = ccc_pgroup_4
        print("ccccccccccccccccccccccccccc")
        print(metrics['group_0_CCC_1'],metrics['group_1_CCC_1'],metrics['group_2_CCC_1'],metrics['group_3_CCC_1'],metrics['group_4_CCC_1'])
        metrics_folds.append(metrics)
    
    mean_metrics = pd.DataFrame(metrics_folds).mean(axis=0)
        
    return mean_metrics


# compute probabilities and predictions for array of noise levels on five folds
def get_results_probas(folds, dataset_info, parameter_settings,attribute):
    
    print('Computing outputs...')
    
    probas =list (parameter_settings['probabilities'])
    
    outputs_probas = {}
    
    for i in probas:
        i = round(i, 5)
        outputs_folds = get_results(folds, i, dataset_info, parameter_settings,attribute)
        outputs_probas[i] = outputs_folds
        
    print('Done.')
    
    return outputs_probas

# compute irr metrics for array of noise levels
def get_metrics_probas(outputs_probas, dataset_info,attribute):
    
    metrics_probas = {}

    for i in outputs_probas:
        print(f'Computing metrics with proba {i} ...')
        mean_metrics = get_metrics(outputs_probas[i], dataset_info,attribute)
        metrics_probas[i] = mean_metrics 
    
    metrics_probas = pd.DataFrame(metrics_probas).T

    print('Done.')

    return metrics_probas

# get irr metrics directly from data
def get_metrics_probas_from_model(dataset_info, parameter_settings, folds,attribute):
    outputs_probas = get_results_probas(folds, dataset_info, parameter_settings,attribute)
    metrics_probas = get_metrics_probas(outputs_probas, dataset_info,attribute)  
    return metrics_probas
 