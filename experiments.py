import train_eval as te
import figures as figs

# auxiliary function to save irr metrics
def save_results(results, parameter_settings):
    results.to_csv(f'results/feature_set={parameter_settings["feature_set"]}_var={parameter_settings["variance"]}_grouped={parameter_settings["grouped"]}_num-min={parameter_settings["num_minima"]}.csv')

# insert features lists from dataset info into parameter settings for experiments
def insert_features_lists(dataset_info, parameter_settings):
    numerical_list = dataset_info['numerical_attributes']
    categorical_list = dataset_info['categorical_attributes']    
    if parameter_settings['feature_set'] == 'all':
        parameter_settings['features_num'] = numerical_list
        parameter_settings['features_cat'] = categorical_list
    elif parameter_settings['feature_set'] == 'num':
        parameter_settings['features_num'] = numerical_list
        parameter_settings['features_cat'] = []
    elif parameter_settings['feature_set'] == 'cat':
        parameter_settings['features_num'] = []
        parameter_settings['features_cat'] = categorical_list
    else: print('Insert appropriate list of features.')
    return parameter_settings

# perform set of experiments as defined by experiment parameters
def experiments(dataset_info, parameter_settings, folds,attribute):
    parameter_settings = parameter_settings.copy()
    insert_features_lists(dataset_info, parameter_settings)
    variance_list = parameter_settings['variance_list']
    if variance_list == None:
        parameter_settings['variance'] = None
        parameter_settings['variance']=None
        metrics_probas = te.get_metrics_probas_from_model(dataset_info, parameter_settings, folds,attribute)
        save_results(metrics_probas, parameter_settings)
        figs.plot_all_figures(metrics_probas, parameter_settings)
    else: 
        for var in variance_list:
            print(f'Computing with variance {var}')
            parameter_settings['variance'] = var
            metrics_probas = te.get_metrics_probas_from_model(dataset_info, parameter_settings, folds,attribute)
            save_results(metrics_probas, parameter_settings)
            print(attribute[0],attribute[1],attribute[2],attribute[3],attribute[4])
            figs.plot_all_figures(metrics_probas, parameter_settings)
    return metrics_probas