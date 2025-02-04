import matplotlib.pyplot as plt
import os
if not os.path.exists('figures1'):
    os.makedirs('figures1')
# five dictionaries with figure type information (kappa, pabak, icc2, bak and pak corrections for two groups)

figure_kappa = {
    'tag': 'Kappa', 
    'group_0_metric': 'group_0_kappa', 
    'group_1_metric': 'group_1_kappa',
    'y_label': "Cohen's Kappa"
}

figure_PABAK = {
    'tag': 'PABAK', 
    'group_0_metric': 'group_0_PABAK', 
    'group_1_metric': 'group_1_PABAK',
    'y_label': "Byrt's PABAK"
}

figure_ICC2 = {
    'tag': 'ICCA', 
    'group_0_metric': 'group_0_ICC_2', 
    'group_1_metric': 'group_1_ICC_2',
    'group_2_metric': 'group_2_ICC_2',
    'group_3_metric': 'group_3_ICC_2',
    'group_4_metric': 'group_4_ICC_2',
    'y_label': "ICCprob(A,1)"
}  
figure_ICC3 = {
    'tag': 'ICCA9', 
    'group_0_metric': 'group_0_ICC_3', 
    'group_1_metric': 'group_1_ICC_3',
    'group_2_metric': 'group_2_ICC_3',
    'group_3_metric': 'group_3_ICC_3',
    'group_4_metric': 'group_4_ICC_3',
    'y_label': "ICCPred(A,1)"
} 
figure_CCC = {
    'tag': 'CCC', 
    'group_0_metric': 'group_0_CCC_1', 
    'group_1_metric': 'group_1_CCC_1',
    'group_2_metric': 'group_2_CCC_1',
    'group_3_metric': 'group_3_CCC_1',
    'group_4_metric': 'group_4_CCC_1',
    'y_label': "CCCProb"
}
figure_correction_group_0 = {
    'tag': 'CORR_G0',
    'group_CK': 'group_0_kappa',
    'group_PABAK': 'group_0_PABAK',
    'group_PAK': 'group_0_PAK',
    'group_BAK': 'group_0_BAK',
    'y_label': 'BI and PI Correction Group 0'
}

figure_correction_group_1 = {
    'tag': 'CORR_G1',
    'group_CK': 'group_1_kappa',
    'group_PABAK': 'group_1_PABAK',
    'group_PAK': 'group_1_PAK',
    'group_BAK': 'group_1_BAK',
    'y_label': 'BI and PI Correction Group 1'
}

# plot and save figures with two metrics (kappa, pabak, icc)
def plot_save_figure_2_metrics(metrics_probas, parameter_settings, figure_type):
    tag = figure_type['tag']
    if tag=='CCC':
      fig = plt.figure()
      subfig = fig.add_subplot(1, 1, 1)
      subfig.plot(metrics_probas[figure_type['group_0_metric']], color = 'blue', alpha = 0.5,   label = 'Africa')
      subfig.plot(metrics_probas[figure_type['group_1_metric']], color = 'orange', label = 'Americas')
      subfig.plot(metrics_probas[figure_type['group_2_metric']], color = 'red', label = 'Asia and Pacific')
      subfig.plot(metrics_probas[figure_type['group_3_metric']], color = 'yellow', label = 'Europe')
      subfig.plot(metrics_probas[figure_type['group_4_metric']], color = 'green', label = 'Mid East')
      subfig.set_ylabel(figure_type['y_label'])
      fig_label = f'feature set={parameter_settings["feature_set"]}, var={parameter_settings["variance"]}, grouped={parameter_settings["grouped"]}, num_min={parameter_settings["num_minima"]}'
      subfig.set_xlabel(f'noise level ({fig_label})')
      subfig.legend()
      fig1 = plt.figure()
      subfig1 = fig1.add_subplot(1, 1, 1)
      subfig1.plot(metrics_probas[figure_type['group_0_metric']], color = 'blue', alpha = 0.5,   label = 'Africa')
      fig_label = f'feature set={parameter_settings["feature_set"]}, var={parameter_settings["variance"]}, grouped={parameter_settings["grouped"]}, num_min={parameter_settings["num_minima"]}'
      subfig.set_xlabel(f'noise level ({fig_label})')
      fig.savefig(f'figu/{tag}_feature_set={parameter_settings["feature_set"]}_var={parameter_settings["variance"]}_grouped={parameter_settings["grouped"]}_num_min={parameter_settings["num_minima"]}_America.png', bbox_inches = "tight") 
      fig1.savefig(f'figu/{tag}_feature_set={parameter_settings["feature_set"]}_var={parameter_settings["variance"]}_grouped={parameter_settings["grouped"]}_num_min={parameter_settings["num_minima"]}.png', bbox_inches = "tight") 
    else:  
      fig = plt.figure()
      subfig = fig.add_subplot(1, 1, 1)
      subfig.plot(metrics_probas[figure_type['group_0_metric']], color = 'blue', alpha = 0.5,   label = 'Africa')
      subfig.plot(metrics_probas[figure_type['group_1_metric']], color = 'orange', label = 'Americas')
      subfig.plot(metrics_probas[figure_type['group_2_metric']], color = 'red', label = 'Asia and Pacific')
      subfig.plot(metrics_probas[figure_type['group_3_metric']], color = 'yellow', label = 'Europe')
      subfig.plot(metrics_probas[figure_type['group_4_metric']], color = 'green', label = 'Mid East')
      subfig.set_ylabel(figure_type['y_label'])
      fig_label = f'feature set={parameter_settings["feature_set"]}, var={parameter_settings["variance"]}, grouped={parameter_settings["grouped"]}, num_min={parameter_settings["num_minima"]}'
      subfig.set_xlabel(f'noise level ({fig_label})')
      subfig.legend()
      fig.savefig(f'figu/{tag}_feature_set={parameter_settings["feature_set"]}_var={parameter_settings["variance"]}_grouped={parameter_settings["grouped"]}_num_min={parameter_settings["num_minima"]}.png', bbox_inches = "tight") 
    
#plot and save figure with four metrics (comparison of kappa and pabak, separately for groups)
def plot_save_figure_corr(metrics_probas, parameter_settings, figure_type):

    return
#plot all figures at once
def plot_all_figures(metrics_probas, parameter_settings):
    for figure_type_2 in [figure_ICC3,figure_CCC]:
        plot_save_figure_2_metrics(metrics_probas, parameter_settings, figure_type_2)
    for figure_type_corr in [figure_correction_group_0, figure_correction_group_1]:
        plot_save_figure_corr(metrics_probas, parameter_settings, figure_type_corr)
