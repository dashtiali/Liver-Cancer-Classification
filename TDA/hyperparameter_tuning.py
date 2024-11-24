# # Experiment with tda features

# ## Settings:
# 1. Whole Liver
# 2. Largest 15 slices
# 3. Slice-wise (images labeled with patient ID and slice number)
# 4. Cubical complex -> barcodes (dim 0,1)
#
# ## Feature Extraction
# 1. Betti Curve
# 2. Entropy Summary
# 3. Pers Landscape
# 4. Pers Stats
# 5. Pers Tropical Coordinates


import pandas as pd
from pycaret.classification import *
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import re
import os
import shutup
from tqdm import tqdm


def aggregate_features(featur_dict):
    feature_agg = []

    for i, feature_name in enumerate(featur_dict):
        for j, dim in enumerate(featur_dict[feature_name]):
            feature_df = pd.read_csv(featur_dict[feature_name][dim])

            # Initialize the aggregated feature data frame with the first two columns
            if i == j == 0:
                feature_agg = pd.DataFrame(feature_df[['Patient_ID', 'File_name']])

            # Drop the first column
            feature_df_mod = feature_df.drop(columns=['Patient_ID'])

            # Merge DataFrames on 'File_name'
            feature_agg = pd.merge(feature_agg, feature_df_mod, on='File_name')

    return feature_agg

def stratified_group_split(data, target_column, group_column, test_size=0.2, random_state=42):
    # Create a DataFrame of unique patient IDs with their class
    unique_patients = data[[group_column, target_column]].drop_duplicates()

    integer_portion = int(unique_patients.shape[0] * test_size)
    test_size = integer_portion/unique_patients.shape[0]

    # Perform the split
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        stratify=unique_patients[target_column],
        random_state=random_state
    )

    # Filter the original DataFrame using train and test patient IDs
    train_df = data[data[group_column].isin(train_patients[group_column])]
    test_df = data[data[group_column].isin(test_patients[group_column])]

    return train_df, test_df

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.selected_features = []

    def fit(self, X, y):
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        self.selected_features = X.columns[lasso.coef_ != 0].tolist()

        return self

    def transform(self, X):
        return X[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features

def save_auc_plot(pycaret_model, file_name):
    plot_model(pycaret_model, plot='auc', save=True, plot_kwargs = {'classes' : get_config('pipeline').steps[0][1].transformer.classes_})

    if os.path.exists('AUC.png'):
        os.rename('AUC.png', file_name)

def save_conf_matrix_plot(pycaret_model, file_name):
    plot_model(pycaret_model, plot='confusion_matrix', save=True, plot_kwargs = {'classes' : get_config('pipeline').steps[0][1].transformer.classes_})

    if os.path.exists('Confusion Matrix.png'):
        os.rename('Confusion Matrix.png', file_name)

def add_space_between_words(string):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', string)

def format_feature_name(feature_name):
    return add_space_between_words(feature_name).replace('Pers', 'Persistent').replace('Stats', 'Statistics')


if __name__ == '__main__':
    shutup.please()

    RS = 123

    main_feat_dir = f'extracted_features'
    output_dir = 'initial_results'
    plots_dir = f'{output_dir}\\plots'
    meta_data_analysis_dir = f'{output_dir}\\meta_data_analysis'
    saved_models_dir = f'{output_dir}\\saved_models'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)

    if not os.path.isdir(meta_data_analysis_dir):
        os.mkdir(meta_data_analysis_dir)

    if not os.path.isdir(saved_models_dir):
        os.mkdir(saved_models_dir)

    feature_sets = [
                        ['BettiCurve'],
                        ['EntropySummary'],
                        ['PersStats'],
                        ['PersLandscape'],
                        ['PersTropicalCoordinates']
                    ]

    ph_dims = [0, 1]

    metric_columns = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']

    folds_result_columns = ['Feature Selection Algorithm',
                            'Feature',
                            'Model',
                            'Accuracy ± Std.',
                            'AUC ± Std.',
                            'Recall ± Std.',
                            'Prec. ± Std.',
                            'F1 ± Std.']

    df_fold_results = pd.DataFrame(columns=folds_result_columns)

    test_result_columns = ['Feature Selection Algorithm', 'Feature', 'Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']
    df_test_results = pd.DataFrame(columns=test_result_columns)

    for feature_list in tqdm(feature_sets):
        feature_name = ' - '.join([format_feature_name(f) for f in feature_list])
        print(f'\nProcessing Feature list {feature_name} ....')
        feature_selection_results = f'Feature: {feature_name}\n'

        feature_dict_ICC = {feature_name:
                            {d: f'{main_feat_dir}\\ICC\\feature_matrix_dim_{d}_{feature_name}.csv' for d in ph_dims}
                        for feature_name in feature_list}

        feature_dict_HCC = {feature_name:
                            {d: f'{main_feat_dir}\\HCC\\feature_matrix_dim_{d}_{feature_name}.csv' for d in ph_dims}
                        for feature_name in feature_list}

        ICC_feature = aggregate_features(feature_dict_ICC)
        HCC_feature = aggregate_features(feature_dict_HCC)

        ICC_feature['Class'] = 'ICC'
        HCC_feature['Class'] = 'HCC'

        combined_data = pd.concat([ICC_feature, HCC_feature], ignore_index=True)

        # Remove columns with all zero values
        combined_data = combined_data.loc[:, (combined_data != 0).any(axis=0)]

        combined_data = combined_data.set_index('File_name')
        train_data, test_data = stratified_group_split(combined_data, target_column='Class', group_column='Patient_ID', test_size=0.2, random_state=RS)

        feature_selectors = {'Lasso': LassoFeatureSelector()}

        for algorithms_name, sel_algorithm in feature_selectors.items():

            # Setup PyCaret
            exp1 = setup(data=train_data,
                         test_data=test_data,
                        target='Class',
                        ignore_features=['Patient_ID'],
                        fold=5,
                        train_size=0.8,
                        fold_strategy="groupkfold",
                        fold_groups='Patient_ID',
                        normalize = True,
                        preprocess=True,
                        normalize_method='zscore',
                        feature_selection=False,
                        custom_pipeline=sel_algorithm,
                        session_id=RS,
                        verbose=False,
                        log_data=False)

            feature_selection_results = f'Feature: {feature_name}\n'
            feature_selection_results += f'Feature selection algorithm: {algorithms_name}\n'
            feature_selection_results += f'Number of selected features: {exp1.get_config("X_train_transformed").shape[1]}\n'
            feature_selection_results += f'Selected features list: [{", ".join(exp1.get_config("X_train_transformed").columns)}]\n\n'

            np.save(f'{output_dir}\\{feature_name}_selected_feature.npy', exp1.get_config("X_train_transformed").columns)

            with open(f"{output_dir}\\feature_selection_results.txt", mode="a") as f:
                f.write(feature_selection_results)

            best_models = compare_models(n_select=3, verbose=False)
            dt_results = pull()
            dt_results.to_csv(f'{output_dir}\\compare_models_{"_".join(feature_list)}.csv', header=True)

            tuned_best_models = []
            models_metrics = []

            for model in best_models:
                print(f'Tuning model: {type(model).__name__} ...')
                tuned_model = tune_model(model, optimize = 'Recall', search_library='optuna', verbose=False)
                tuned_best_models.append(tuned_model)
                dt_results = pull()
                models_metrics.append(dt_results)
                save_model(model, f'{saved_models_dir}\\{feature_name}_tuned_{type(model).__name__}_model',)

            model_names = [add_space_between_words(type(model).__name__) for model in tuned_best_models]

            data = []

            for model_name, metrics in zip(model_names, models_metrics):
                formatted_metrics = [algorithms_name, feature_name, model_name]
                mean_values = metrics.loc['Mean', metric_columns]
                std_values = metrics.loc['Std', metric_columns]
                formatted_metrics.extend([f"{mean:.4f} ± {std:.4f}" for mean, std in zip(mean_values, std_values)])
                data.append(formatted_metrics)

            df = pd.DataFrame(data, columns=folds_result_columns)
            df.to_csv(f'{output_dir}\\tuned_models_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True, encoding="cp1252")
            df_fold_results = pd.concat([df_fold_results, df], ignore_index=True)

            # Make predictions on test data
            #######################################

            df = pd.DataFrame(columns=test_result_columns)

            for model_name, model in zip(model_names, tuned_best_models):
                try:
                    save_auc_plot(model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_{model_name}_model_test_data_auc_plot.png')
                    save_conf_matrix_plot(model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_{model_name}_model_test_data_confusion_matrix.png')
                except:
                    print(f'Cannot create all the plots for {model_name} !\n')

                predictions = predict_model(model, verbose=False)
                results = pull()
                results['Feature Selection Algorithm'] = algorithms_name
                results['Feature'] = feature_name
                results['Model'] = model_name

                df = pd.concat([df, results[test_result_columns]], ignore_index=True)


            df.to_csv(f'{output_dir}\\tuned_best_models_test_data_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True)
            df_test_results = pd.concat([df_test_results, df], ignore_index=True)

            #######################################

            print(f'Creating ensemble stack models ...')
            stack_model = stack_models(estimator_list = tuned_best_models.copy(), meta_model = tuned_best_models[0], optimize = 'Recall', verbose=False)

            print(f'Tuning ensemble model ...')
            tuned_stack_model = tune_model(stack_model)
            tuned_stack_model_results = pull()

            formatted_metrics = [algorithms_name, feature_name, 'Ensemble']
            mean_values = tuned_stack_model_results.loc['Mean', metric_columns]
            std_values = tuned_stack_model_results.loc['Std', metric_columns]
            formatted_metrics.extend([f"{mean:.4f} ± {std:.2f}" for mean, std in zip(mean_values, std_values)])

            df = pd.DataFrame([formatted_metrics], columns=folds_result_columns)
            df.to_csv(f'{output_dir}\\tuned_ensemble_model_avg_5_fold_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True, encoding="cp1252")
            df_fold_results = pd.concat([df_fold_results, df], ignore_index=True)

            # Make predictions on test data
            #######################################

            try:
                save_auc_plot(tuned_stack_model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_ensemble_model_test_data_auc_plot.png')
                save_conf_matrix_plot(tuned_stack_model, f'{plots_dir}\\{"_".join(feature_list)}_{algorithms_name}_tuned_ensemble_model_test_data_confusion_matrix.png')
            except:
                print(f'Cannot create all the plots for ensemble model!\n')

            predictions = predict_model(tuned_stack_model)
            tuned_stack_model_results = pull()

            tuned_stack_model_results['Feature Selection Algorithm'] = algorithms_name
            tuned_stack_model_results['Feature'] = feature_name
            tuned_stack_model_results['Model'] = model_name

            df = tuned_stack_model_results[test_result_columns]
            df.to_csv(f'{output_dir}\\tuned_ensemble_model_test_data_metrics_{"_".join(feature_list)}_{algorithms_name}.csv', header=True)


            df_test_results = pd.concat([df_test_results, df], ignore_index=True)
            save_model(tuned_stack_model, f'{saved_models_dir}\\{feature_name}_tuned_ensemble_model',)

            #######################################

            try:
                get_patient_ID = lambda x: re.match(r'^(.*?)_slice+', x).group(1)
                predictions['File_name'] = predictions.index
                predictions.reset_index(drop=True, inplace=True)
                predictions['Patient_ID'] = predictions['File_name'].apply(get_patient_ID)
                predictions = predictions[['Patient_ID', 'File_name', 'Class', 'prediction_label', 'prediction_score']]

                predictions.to_excel(f'{meta_data_analysis_dir}\\{"_".join(feature_list)}_{algorithms_name}_meta_analysis_data.xlsx', index=False)
            except:
                pass

    df_fold_results.to_csv(f'{output_dir}\\all_fold_results.csv', header=True, encoding="cp1252")
    df_test_results.to_csv(f'{output_dir}\\all_test_results.csv', header=True, encoding="cp1252")
