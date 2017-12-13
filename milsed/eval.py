# CREATED: 7/25/17 15:35 by Justin Salamon <justin.salamon@nyu.edu>

import sklearn
from tqdm import tqdm
import sed_eval
import os
import milsed
import pumpp
import numpy as np
import jams
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import OrderedDict


def score_model(OUTPUT_PATH, pump, model, idx, pumpfolder, labelfile, duration,
                version, use_tqdm=False, use_orig_duration=False,
                save_jams=True, weak_from_strong=False):

    results = {}

    # For computing weak metrics
    weak_true = []
    weak_pred = []

    # For computing strong (sed_eval) metrics
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        pump['static'].encoder.classes_.tolist(), time_resolution=1.0)

    if weak_from_strong:
        pred_folder = os.path.join(OUTPUT_PATH, version,
                                   'predictions_weakfromstrong')
    else:
        pred_folder = os.path.join(OUTPUT_PATH, version, 'predictions')

    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)

    # Predict on test, file by file, and compute eval scores
    if use_tqdm:
        idx = tqdm(idx, desc='Evaluating the model')

    # Load durations filel
    if use_orig_duration:
        durfile = os.path.join(OUTPUT_PATH, 'durations.json')
        durations = json.load(open(durfile, 'r'))

    for fid in idx:

        # Load test data
        pumpfile = os.path.join(pumpfolder, fid + '.h5')
        dpump = milsed.utils.load_h5(pumpfile)
        datum = dpump['mel/mag']
        ytrue = dpump['static/tags'][0]

        # Predict
        output_d, output_s = model.predict(datum)

        # If output is smaller in time dimension that input, interpolate
        if output_d.shape[1] != datum.shape[1]:
            output_d = milsed.utils.interpolate_prediction(output_d, duration,
                                                           datum.shape[1])

        # Append weak predictions
        if weak_from_strong:
            wfs_pred = np.max(output_d[0], axis=0)
            weak_pred.append((wfs_pred >= 0.5) * 1)
        else:
            weak_pred.append((output_s[0] >= 0.5) * 1)  # binarize
        weak_true.append(ytrue * 1)  # convert from bool to int

        # Build a dynamic task label transformer for the strong predictions
        dynamic_trans = pumpp.task.DynamicLabelTransformer(
            name='dynamic', namespace='tag_open',
            labels=pump['static'].encoder.classes_,
            sr=pump['mel'].sr,
            hop_length=pump['mel'].hop_length)
        dynamic_trans.encoder = pump['static'].encoder

        # Convert weak and strong predictions into JAMS annotations
        ann_s = pump['static'].inverse(output_s[0], duration=duration)
        ann_d = dynamic_trans.inverse(output_d[0], duration=duration)

        # add basic annotation metadata
        ann_s.annotation_metadata.version = version
        ann_s.annotation_metadata.annotation_tools = 'static'
        ann_d.annotation_metadata.version = version
        ann_d.annotation_metadata.annotation_tools = 'dynamic'

        # Create reference jams annotation
        ref_jam = milsed.utils.create_dcase_jam(fid, labelfile,
                                                duration=duration,
                                                weak=False)
        ann_r = ref_jam.annotations.search(annotation_tools='reference')[0]

        # Add annotations to jams
        jam = jams.JAMS()
        jam.annotations.append(ann_s)
        jam.annotations.append(ann_d)
        jam.annotations.append(ann_r)

        # file metadata
        jam.file_metadata.duration = duration
        jam.file_metadata.title = fid

        # Trim annotations to original file's duration
        if use_orig_duration:
            orig_duration = durations[fid]
            jam = jam.trim(0, orig_duration, strict=False)
            ann_s = jam.annotations.search(annotation_tools='static')[0]
            ann_d = jam.annotations.search(annotation_tools='dynamic')[0]
            ann_r = jam.annotations.search(annotation_tools='reference')[0]

        if save_jams:
            jamfile = os.path.join(pred_folder, '{:s}.jams'.format(fid))
            jam.save(jamfile)

        # Compute intermediate stats for sed_eval metrics
        # sed_eval expects a list containing a dict for each event, where the
        # dict keys are event_onset, event_offset, event_label.
        ref_list = []
        for event in ann_r.data:
            ref_list.append({'event_onset': event.time,
                             'event_offset': event.time + event.duration,
                             'event_label': event.value})
        ref_list = sed_eval.util.event_list.EventList(ref_list)

        est_list = []
        for event in ann_d.data:
            est_list.append({'event_onset': event.time,
                             'event_offset': event.time + event.duration,
                             'event_label': event.value})
        est_list = sed_eval.util.event_list.EventList(est_list)

        segment_based_metrics.evaluate(ref_list, est_list)

    # Compute weak metrics
    weak_true = np.asarray(weak_true)
    weak_pred = np.asarray(weak_pred)
    weak_pred = (weak_pred >= 0.5) * 1  # binarize

    results['weak'] = {}
    for avg in ['micro', 'macro', 'weighted', 'samples']:
        results['weak'][avg] = {}
        results['weak'][avg]['f1'] = sklearn.metrics.f1_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['precision'] = sklearn.metrics.precision_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['recall'] = sklearn.metrics.recall_score(
            weak_true, weak_pred, average=avg)

    # results['weak']['f1_micro'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='micro')
    # results['weak']['f1_macro'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='macro')
    # results['weak']['f1_weighted'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='weighted')
    # results['weak']['f1_samples'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='samples')

    # Compute strong (sed_eval) metrics
    results['strong'] = segment_based_metrics.results()

    return results


def report_results(OUTPUT_PATH, version):
    # Load results
    resultsfolder = os.path.join(OUTPUT_PATH, version)
    resultsfile = os.path.join(resultsfolder, 'results.json')
    with open(resultsfile, 'r') as fp:
        results = json.load(fp)

    # report
    print('{:<10}{}'.format('Model', version))
    print('\nWeak:')
    for metric in results['weak']['micro'].keys():
        print('{:<10}{:.3f}'.format(metric, results['weak']['micro'][metric]))

    print('\nStrong:')
    strong_f = results['strong']['overall']['f_measure']
    strong_e = results['strong']['overall']['error_rate']
    print('{:<10}{:.3f}'.format('precision', strong_f['precision']))
    print('{:<10}{:.3f}'.format('recall', strong_f['recall']))
    print('{:<10}{:.3f}'.format('f1', strong_f['f_measure']))
    print('{:<10}{:.3f}'.format('e_rate', strong_e['error_rate']))

    print('\n{:<40}P\tR\tF\tE'.format('Strong per-class:'))
    strong_c = results['strong']['class_wise']
    c_sorted = [c for c in strong_c.keys()]
    c_sorted = sorted(c_sorted)
    for c in c_sorted:
        r_c = strong_c[c]['f_measure']
        r_ce = strong_c[c]['error_rate']
        print('{:<40}{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(c, r_c['precision'],
                                                            r_c['recall'],
                                                            r_c['f_measure'],
                                                            r_ce['error_rate']))

    # # Load training history
    # history_file = os.path.join(resultsfolder, 'history.pkl')
    # with open(history_file, 'rb') as fp:
    #     history = pickle.load(fp)

    # Load dynamic history CSV file
    csvfile = os.path.join(resultsfolder, 'history_csvlog.csv')
    history = pd.read_csv(csvfile)

    # Set sns style
    sns.set()

    print('\nLoss:')

    # Visualize training history
    plt.subplot(1,2,1)
    plt.plot(history['loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss: {}'.format(version))
    # plt.grid()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history['static/tags_acc'], label='training accuracy')
    plt.plot(history['val_static/tags_acc'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy: {}'.format(version))
    plt.tight_layout()

    plt.show()


def compare_results(OUTPUT_PATH, versions, sort=False, weak_from_strong=False,
                    is_ensemble=False):
    results = OrderedDict({})
    params = OrderedDict({})
    n_weights = OrderedDict({})

    # Load pump
    pump = pickle.load(
        open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb'))

    # Load results
    for version in versions:

        # Load results
        if weak_from_strong:
            resultsfile = os.path.join(OUTPUT_PATH, version,
                                       'predictions_weakfromstrong',
                                       'results.json')
        else:
            resultsfile = os.path.join(OUTPUT_PATH, version, 'results.json')
        with open(resultsfile, 'r') as fp:
            results[version] = json.load(fp)

        if is_ensemble:
            n_weights[version] = 'ensemble'
            params[version] = {'modelname': version}
        else:
            # Load params
            paramsfile = os.path.join(OUTPUT_PATH, version, 'params.json')
            with open(paramsfile, 'r') as fp:
                params[version] = json.load(fp)

            # Compute model size
            model, _, _ = milsed.models.MODELS[params[version]['modelname']](
                pump, params[version]['alpha'])
            n_weights[version] = model.count_params()

    # Convert to dataframe
    df = pd.DataFrame(
        columns=['version', 'model', 'n_weights', 'w_f1(ma)', 'w_f1(mi)',
                 'w_p(mi)', 'w_r(mi)', 's_f1', 's_p', 's_r', 's_e'])
    for k in results.keys():
        r = results[k]
        weak_macro = r['weak']['macro']
        weak = r['weak']['micro']
        strong_f = r['strong']['overall']['f_measure']
        strong_e = r['strong']['overall']['error_rate']
        data = (
            k, params[k]['modelname'], n_weights[k], weak_macro['f1'],
            weak['f1'], weak['precision'], weak['recall'],
            strong_f['f_measure'], strong_f['precision'], strong_f['recall'],
            strong_e['error_rate'])
        df.loc[len(df), :] = data

    if sort:
        df = df.sort_values('version')
    return df


def predict_eval(OUTPUT_PATH, pump, model_list, idx, pumpfolder, duration,
                 version, use_tqdm=False, use_orig_duration=True,
                 save_jams=True, weak_from_strong=False,
                 using_test_set=False):
    '''
    Predict on EVAL SET and store predictions.

    Parameters
    ----------
    OUTPUT_PATH
    pump
    model
    idx
    pumpfolder
    duration
    version
    use_tqdm
    use_orig_duration
    save_jams

    Returns
    -------

    '''
    def geo_mean(data, axis=0, use_log=True):
        if use_log:
            return np.exp(np.log(data).sum(axis=axis) / len(data))
        else:
            return data.prod(axis=axis)**(1.0/len(data))

    # For storing predictions across all eval files
    df_d_all = pd.DataFrame(
        columns=['filename', 'start_time', 'end_time', 'label'])
    df_s_all = pd.DataFrame(
        columns=['filename', 'start_time', 'end_time', 'label'])

    # Create folder for predictions
    if weak_from_strong:
        pred_folder = os.path.join(OUTPUT_PATH, version,
                                   'predictions_eval_weakfromstrong')
    else:
        pred_folder = os.path.join(OUTPUT_PATH, version, 'predictions_eval')

    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)

    # Predict on test, file by file, and compute eval scores
    if use_tqdm:
        idx = tqdm(idx, desc='Evaluating the model')

    # Load durations file
    if use_orig_duration:
        durfile = os.path.join(OUTPUT_PATH, 'durations.json')
        durations = json.load(open(durfile, 'r'))

    for fid in idx:

        # Load eval data
        pumpfile = os.path.join(pumpfolder, fid + '.h5')
        dpump = milsed.utils.load_h5(pumpfile)
        datum = dpump['mel/mag']
        ytrue = dpump['static/tags'][0]  # dummy data

        # Predict
        if len(model_list)==1:
            output_d, output_s = model_list[0].predict(datum)
        else:
            # Predict from every model
            output_d_ensemble = []
            output_s_ensemble = []
            for model in model_list:
                output_d, output_s = model.predict(datum)
                output_d_ensemble.append(output_d[0])
                output_s_ensemble.append(output_s[0])

            # Late fusion using geometric mean
            output_d_ensemble = np.asarray(output_d_ensemble)
            output_s_ensemble = np.asarray(output_s_ensemble)

            output_d = geo_mean(output_d_ensemble, axis=0, use_log=True)
            output_s = geo_mean(output_s_ensemble, axis=0, use_log=True)

            output_d = np.expand_dims(output_d, axis=0)
            output_s = np.expand_dims(output_s, axis=0)

        # If output is smaller in time dimension that input, interpolate
        if output_d.shape[1] != datum.shape[1]:
            output_d = milsed.utils.interpolate_prediction(output_d, duration,
                                                           datum.shape[1])

        # Create weak from strong
        if weak_from_strong:
            wfs_pred = np.max(output_d[0], axis=0)
            output_s[0] = (wfs_pred >= 0.5) * 1

        # Build a dynamic task label transformer for the strong predictions
        dynamic_trans = pumpp.task.DynamicLabelTransformer(
            name='dynamic', namespace='tag_open',
            labels=pump['static'].encoder.classes_)
        dynamic_trans.encoder = pump['static'].encoder

        # Convert weak and strong predictions into JAMS annotations
        ann_s = pump['static'].inverse(output_s[0], duration=duration)
        ann_d = dynamic_trans.inverse(output_d[0], duration=duration)

        # add basic annotation metadata
        ann_s.annotation_metadata.version = version
        ann_s.annotation_metadata.annotation_tools = 'static'
        ann_d.annotation_metadata.version = version
        ann_d.annotation_metadata.annotation_tools = 'dynamic'

        # Add annotations to jams
        jam = jams.JAMS()
        jam.file_metadata.duration = duration
        jam.file_metadata.title = fid

        jam.annotations.append(ann_s)
        jam.annotations.append(ann_d)

        # Trim annotations to original file's duration
        if use_orig_duration:
            orig_duration = durations[fid]
            jam = jam.trim(0, orig_duration, strict=False)
            ann_s = jam.annotations.search(annotation_tools='static')[0]
            ann_d = jam.annotations.search(annotation_tools='dynamic')[0]

        if save_jams:
            jamfile = os.path.join(pred_folder, '{:s}.jams'.format(fid))
            jam.save(jamfile)

        # Append weak and strong results to overall dataframes

        # PROCESS DYNAMIC LABELS
        df_d = ann_d.to_dataframe()
        # df_d['filename'] = 'audio/{}.wav'.format(fid) # example on web is bad
        df_d['filename'] = '{}.wav'.format(fid[1:])
        df_d['start_time'] = df_d.time
        df_d['end_time'] = df_d.time + df_d.duration
        df_d['label'] = df_d['value']
        df_d_ordered = df_d[['filename', 'start_time', 'end_time', 'label']]

        df_d_all = df_d_all.append(df_d_ordered)

        # PROCESS STATIC LABELS
        df_s = ann_s.to_dataframe()
        # df_s['filename'] = 'audio/{}.wav'.format(fid) # example on web is bad
        df_s['filename'] = '{}.wav'.format(fid[1:])
        df_s['start_time'] = df_s.time
        df_s['end_time'] = df_s.time + df_s.duration
        df_s['label'] = df_s['value']
        df_s_ordered = df_s[['filename', 'start_time', 'end_time', 'label']]

        df_s_all = df_s_all.append(df_s_ordered)

    # MUST ADD EMPTY ROWS FOR FILES WITH NO PREDICTIONS
    ## STATIC ##
    added_s = 0
    uniquefiles_s = np.unique(
        ['Y'+v.replace('.wav', '') for v in
         df_s_all.filename.values])

    missing_files_s = []
    for filename in idx:
        if filename not in uniquefiles_s:
            # df = pd.DataFrame(
            #     columns=['filename', 'start_time', 'end_time', 'label'])
            # df.loc[-1, 'filename'] = 'audio/{}.wav'.format(filename)
            # df_s_all = df_s_all.append(df)
            missing_files_s.append(filename)
            added_s += 1

    ## DYNAMIC ##
    added_d = 0
    uniquefiles_d = np.unique(
        ['Y'+v.replace('.wav', '') for v in
         df_d_all.filename.values])

    missing_files_d = []
    for filename in idx:
        if filename not in uniquefiles_d:
            # df = pd.DataFrame(
            #     columns=['filename', 'start_time', 'end_time', 'label'])
            # df.loc[-1, 'filename'] = 'audio/{}.wav'.format(filename)
            # df_d_all = df_d_all.append(df)
            missing_files_d.append(filename)
            added_d += 1

    # Save results to disk
    if using_test_set:
        dfile = os.path.join(pred_folder, 'pred_test_strong.txt')
        sfile = os.path.join(pred_folder, 'pred_test_weak.txt')
    else:
        dfile = os.path.join(pred_folder, 'pred_eval_strong.txt')
        sfile = os.path.join(pred_folder, 'pred_eval_weak.txt')

    # Write to disk
    df_d_all.to_csv(dfile, header=False, index=False, sep='\t')
    df_s_all.to_csv(sfile, header=False, index=False, sep='\t')

    # Add files with no preds (DCASE format)
    # STATIC
    with open(sfile, "a") as myfile:
        for n, x in enumerate(missing_files_s):
            if n + 1 == len(missing_files_s):
                # myfile.write('audio/{}.wav'.format(x))
                myfile.write('{}.wav'.format(x[1:]))
            else:
                # myfile.write('audio/{}.wav\n'.format(x))
                myfile.write('{}.wav\n'.format(x[1:]))

    # DYNAMIC
    with open(dfile, "a") as myfile:
        for n, x in enumerate(missing_files_d):
            if n + 1 == len(missing_files_d):
                # myfile.write('audio/{}.wav'.format(x))
                myfile.write('{}.wav'.format(x[1:]))
            else:
                # myfile.write('audio/{}.wav\n'.format(x))
                myfile.write('{}.wav\n'.format(x[1:]))

    # Return
    return df_s_all, df_d_all


def score_ensemble(OUTPUT_PATH, pump, model_list, idx, pumpfolder, labelfile,
                   duration,
                   ensemble_version, use_tqdm=False, use_orig_duration=False,
                   save_jams=True, weak_from_strong=False):

    def geo_mean(data, axis=0, use_log=True):
        if use_log:
            return np.exp(np.log(data).sum(axis=axis) / len(data))
        else:
            return data.prod(axis=axis)**(1.0/len(data))

    results = {}

    # For computing weak metrics
    weak_true = []
    weak_pred = []

    # For computing strong (sed_eval) metrics
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        pump['static'].encoder.classes_.tolist(), time_resolution=1.0)

    # Create folder for predictions
    ensemble_folder = os.path.join(OUTPUT_PATH, ensemble_version)
    if not os.path.isdir(ensemble_folder):
        os.mkdir(ensemble_folder)

    if weak_from_strong:
        pred_folder = os.path.join(OUTPUT_PATH, ensemble_version,
                                   'predictions_weakfromstrong')
    else:
        pred_folder = os.path.join(OUTPUT_PATH, ensemble_version,
                                   'predictions')

    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)

    # Predict on test, file by file, and compute eval scores
    if use_tqdm:
        idx = tqdm(idx, desc='Evaluating the ensemble')

    # Load durations filel
    if use_orig_duration:
        durfile = os.path.join(OUTPUT_PATH, 'durations.json')
        durations = json.load(open(durfile, 'r'))

    for fid in idx:

        # Load test data
        pumpfile = os.path.join(pumpfolder, fid + '.h5')
        dpump = milsed.utils.load_h5(pumpfile)
        datum = dpump['mel/mag']
        ytrue = dpump['static/tags'][0]

        # Predict from every model
        output_d_ensemble = []
        output_s_ensemble = []
        for model in model_list:
            output_d, output_s = model.predict(datum)
            output_d_ensemble.append(output_d[0])
            output_s_ensemble.append(output_s[0])

        # Late fusion using geometric mean
        output_d_ensemble = np.asarray(output_d_ensemble)
        output_s_ensemble = np.asarray(output_s_ensemble)

        output_d = geo_mean(output_d_ensemble, axis=0, use_log=True)
        output_s = geo_mean(output_s_ensemble, axis=0, use_log=True)

        output_d = np.expand_dims(output_d, axis=0)
        output_s = np.expand_dims(output_s, axis=0)

        # If output is smaller in time dimension that input, interpolate
        if output_d.shape[1] != datum.shape[1]:
            output_d = milsed.utils.interpolate_prediction(output_d, duration,
                                                           datum.shape[1])

        # Append weak predictions
        if weak_from_strong:
            wfs_pred = np.max(output_d[0], axis=0)
            weak_pred.append((wfs_pred >= 0.5)*1)
        else:
            weak_pred.append((output_s[0]>=0.5)*1)  # binarize
        weak_true.append(ytrue * 1)  # convert from bool to int

        # Build a dynamic task label transformer for the strong predictions
        dynamic_trans = pumpp.task.DynamicLabelTransformer(
            name='dynamic', namespace='tag_open',
            labels=pump['static'].encoder.classes_)
        dynamic_trans.encoder = pump['static'].encoder

        # Convert weak and strong predictions into JAMS annotations
        ann_s = pump['static'].inverse(output_s[0], duration=duration)
        ann_d = dynamic_trans.inverse(output_d[0], duration=duration)

        # add basic annotation metadata
        ann_s.annotation_metadata.version = ensemble_version
        ann_s.annotation_metadata.annotation_tools = 'static'
        ann_d.annotation_metadata.version = ensemble_version
        ann_d.annotation_metadata.annotation_tools = 'dynamic'

        # Create reference jams annotation
        ref_jam = milsed.utils.create_dcase_jam(fid, labelfile, duration=10.0,
                                                weak=False)
        ann_r = ref_jam.annotations.search(annotation_tools='reference')[0]

        # Add annotations to jams
        jam = jams.JAMS()
        jam.annotations.append(ann_s)
        jam.annotations.append(ann_d)
        jam.annotations.append(ann_r)

        # file metadata
        jam.file_metadata.duration = duration
        jam.file_metadata.title = fid

        # Trim annotations to original file's duration
        if use_orig_duration:
            orig_duration = durations[fid]
            jam = jam.trim(0, orig_duration, strict=False)
            ann_s = jam.annotations.search(annotation_tools='static')[0]
            ann_d = jam.annotations.search(annotation_tools='dynamic')[0]
            ann_r = jam.annotations.search(annotation_tools='reference')[0]

        if save_jams:
            jamfile = os.path.join(pred_folder, '{:s}.jams'.format(fid))
            jam.save(jamfile)

        # Compute intermediate stats for sed_eval metrics
        # sed_eval expects a list containing a dict for each event, where the
        # dict keys are event_onset, event_offset, event_label.
        ref_list = []
        for event in ann_r.data:
            ref_list.append({'event_onset': event.time,
                             'event_offset': event.time + event.duration,
                             'event_label': event.value})
        ref_list = sed_eval.util.event_list.EventList(ref_list)

        est_list = []
        for event in ann_d.data:
            est_list.append({'event_onset': event.time,
                             'event_offset': event.time + event.duration,
                             'event_label': event.value})
        est_list = sed_eval.util.event_list.EventList(est_list)

        segment_based_metrics.evaluate(ref_list, est_list)

    # Compute weak metrics
    weak_true = np.asarray(weak_true)
    weak_pred = np.asarray(weak_pred)
    weak_pred = (weak_pred >= 0.5) * 1  # binarize

    results['weak'] = {}
    for avg in ['micro', 'macro', 'weighted', 'samples']:
        results['weak'][avg] = {}
        results['weak'][avg]['f1'] = sklearn.metrics.f1_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['precision'] = sklearn.metrics.precision_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['recall'] = sklearn.metrics.recall_score(
            weak_true, weak_pred, average=avg)

    # results['weak']['f1_micro'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='micro')
    # results['weak']['f1_macro'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='macro')
    # results['weak']['f1_weighted'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='weighted')
    # results['weak']['f1_samples'] = sklearn.metrics.f1_score(
    #     weak_true, weak_pred, average='samples')

    # Compute strong (sed_eval) metrics
    results['strong'] = segment_based_metrics.results()

    return results


def score_ensemble_wfs(
        OUTPUT_PATH, pump, model_list, idx, pumpfolder, labelfile,
        duration,
        ensemble_version, use_tqdm=False, use_orig_duration=True,
        save_jams=True):

    def geo_mean(data, axis=0, use_log=True):
        if use_log:
            return np.exp(np.log(data).sum(axis=axis) / len(data))
        else:
            return data.prod(axis=axis)**(1.0/len(data))

    results = {}
    results_wfs = {}

    # For computing weak metrics
    weak_true = []
    weak_pred = []
    weak_pred_wfs = []

    # For computing strong (sed_eval) metrics
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        pump['static'].encoder.classes_.tolist(), time_resolution=1.0)

    # Create folder for predictions
    ensemble_folder = os.path.join(OUTPUT_PATH, ensemble_version)
    if not os.path.isdir(ensemble_folder):
        os.mkdir(ensemble_folder)


    pred_folder = os.path.join(OUTPUT_PATH, ensemble_version, 'predictions')
    pred_folder_wfs = os.path.join(OUTPUT_PATH, ensemble_version,
                                   'predictions_weakfromstrong')

    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)
    if not os.path.isdir(pred_folder_wfs):
        os.mkdir(pred_folder_wfs)

    # Predict on test, file by file, and compute eval scores
    if use_tqdm:
        idx = tqdm(idx, desc='Evaluating the ensemble')

    # Load durations filel
    if use_orig_duration:
        durfile = os.path.join(OUTPUT_PATH, 'durations.json')
        durations = json.load(open(durfile, 'r'))

    for fid in idx:

        # Load test data
        pumpfile = os.path.join(pumpfolder, fid + '.h5')
        dpump = milsed.utils.load_h5(pumpfile)
        datum = dpump['mel/mag']
        ytrue = dpump['static/tags'][0]

        # Predict from every model
        output_d_ensemble = []
        output_s_ensemble = []
        for model in model_list:
            output_d, output_s = model.predict(datum)
            output_d_ensemble.append(output_d[0])
            output_s_ensemble.append(output_s[0])

        # Late fusion using geometric mean
        output_d_ensemble = np.asarray(output_d_ensemble)
        output_s_ensemble = np.asarray(output_s_ensemble)

        output_d = geo_mean(output_d_ensemble, axis=0, use_log=True)
        output_s = geo_mean(output_s_ensemble, axis=0, use_log=True)

        output_d = np.expand_dims(output_d, axis=0)
        output_s = np.expand_dims(output_s, axis=0)

        # If output is smaller in time dimension that input, interpolate
        if output_d.shape[1] != datum.shape[1]:
            output_d = milsed.utils.interpolate_prediction(output_d, duration,
                                                           datum.shape[1])

        # Append weak predictions
        weak_pred.append((output_s[0] >= 0.5) * 1)  # binarize
        wfs_pred = np.max(output_d[0], axis=0)
        weak_pred_wfs.append((wfs_pred >= 0.5)*1)

        weak_true.append(ytrue * 1)  # convert from bool to int

        # Build a dynamic task label transformer for the strong predictions
        dynamic_trans = pumpp.task.DynamicLabelTransformer(
            name='dynamic', namespace='tag_open',
            labels=pump['static'].encoder.classes_)
        dynamic_trans.encoder = pump['static'].encoder

        # Convert weak and strong predictions into JAMS annotations
        ann_s = pump['static'].inverse(output_s[0], duration=duration)
        ann_s_wfs = pump['static'].inverse(wfs_pred, duration=duration)
        ann_d = dynamic_trans.inverse(output_d[0], duration=duration)

        # add basic annotation metadata
        ann_s.annotation_metadata.version = ensemble_version
        ann_s_wfs.annotation_metadata.version = ensemble_version
        ann_s.annotation_metadata.annotation_tools = 'static'
        ann_s_wfs.annotation_metadata.annotation_tools = 'static'
        ann_d.annotation_metadata.version = ensemble_version
        ann_d.annotation_metadata.annotation_tools = 'dynamic'

        # Create reference jams annotation
        ref_jam = milsed.utils.create_dcase_jam(fid, labelfile, duration=10.0,
                                                weak=False)
        ann_r = ref_jam.annotations.search(annotation_tools='reference')[0]

        # Add annotations to jams
        jam = jams.JAMS()
        jam.annotations.append(ann_s)
        jam.annotations.append(ann_d)
        jam.annotations.append(ann_r)

        # file metadata
        jam.file_metadata.duration = duration
        jam.file_metadata.title = fid

        # Add annotations to jams
        jam_wfs = jams.JAMS()
        jam_wfs.annotations.append(ann_s_wfs)
        jam_wfs.annotations.append(ann_d)
        jam_wfs.annotations.append(ann_r)

        # file metadata
        jam_wfs.file_metadata.duration = duration
        jam_wfs.file_metadata.title = fid

        # Trim annotations to original file's duration
        if use_orig_duration:
            orig_duration = durations[fid]
            jam = jam.trim(0, orig_duration, strict=False)
            ann_s = jam.annotations.search(annotation_tools='static')[0]
            ann_d = jam.annotations.search(annotation_tools='dynamic')[0]
            ann_r = jam.annotations.search(annotation_tools='reference')[0]

            jam_wfs = jam_wfs.trim(0, orig_duration, strict=False)
            ann_s_wfs = jam_wfs.annotations.search(annotation_tools='static')[0]

        if save_jams:
            jamfile = os.path.join(pred_folder, '{:s}.jams'.format(fid))
            jam.save(jamfile)

            jamfile_wfs = os.path.join(pred_folder_wfs, '{:s}.jams'.format(fid))
            jam_wfs.save(jamfile_wfs)

        # Compute intermediate stats for sed_eval metrics
        # sed_eval expects a list containing a dict for each event, where the
        # dict keys are event_onset, event_offset, event_label.
        ref_list = []
        for event in ann_r.data:
            ref_list.append({'event_onset': event.time,
                             'event_offset': event.time + event.duration,
                             'event_label': event.value})
        ref_list = sed_eval.util.event_list.EventList(ref_list)

        est_list = []
        for event in ann_d.data:
            est_list.append({'event_onset': event.time,
                             'event_offset': event.time + event.duration,
                             'event_label': event.value})
        est_list = sed_eval.util.event_list.EventList(est_list)

        segment_based_metrics.evaluate(ref_list, est_list)

    # Compute weak metrics
    weak_true = np.asarray(weak_true)
    weak_pred = np.asarray(weak_pred)
    weak_pred = (weak_pred >= 0.5) * 1  # binarize
    weak_pred_wfs = np.asarray(weak_pred_wfs)
    weak_pred_wfs = (weak_pred_wfs >= 0.5) * 1  # binarize

    results['weak'] = {}
    for avg in ['micro', 'macro', 'weighted', 'samples']:
        results['weak'][avg] = {}
        results['weak'][avg]['f1'] = sklearn.metrics.f1_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['precision'] = sklearn.metrics.precision_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['recall'] = sklearn.metrics.recall_score(
            weak_true, weak_pred, average=avg)

    results_wfs['weak'] = {}
    for avg in ['micro', 'macro', 'weighted', 'samples']:
        results_wfs['weak'][avg] = {}
        results_wfs['weak'][avg]['f1'] = sklearn.metrics.f1_score(
            weak_true, weak_pred_wfs, average=avg)
        results_wfs['weak'][avg]['precision'] = sklearn.metrics.precision_score(
            weak_true, weak_pred_wfs, average=avg)
        results_wfs['weak'][avg]['recall'] = sklearn.metrics.recall_score(
            weak_true, weak_pred_wfs, average=avg)

    # Compute strong (sed_eval) metrics
    results['strong'] = segment_based_metrics.results()
    results_wfs['strong'] = segment_based_metrics.results()

    return results, results_wfs


def score_model_validation(
        OUTPUT_PATH, pump, model, idx, pumpfolder, labelfile, duration,
        version, use_tqdm=False, use_orig_duration=False,
        save_jams=True, weak_from_strong=False):

    results = {}

    # For computing weak metrics
    weak_true = []
    weak_pred = []

    # For computing strong (sed_eval) metrics
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        pump['static'].encoder.classes_.tolist(), time_resolution=1.0)

    pred_folder = os.path.join(OUTPUT_PATH, version, 'predictions_validation')
    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)

    if weak_from_strong:
        pred_folder = os.path.join(pred_folder, 'predictions_weakfromstrong')
    else:
        pred_folder = os.path.join(pred_folder, 'predictions')

    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)

    # Predict on test, file by file, and compute eval scores
    if use_tqdm:
        idx = tqdm(idx, desc='Evaluating the model')

    # Load durations filel
    if use_orig_duration:
        durfile = os.path.join(OUTPUT_PATH, 'durations.json')
        durations = json.load(open(durfile, 'r'))

    for fid in idx:

        # Load test data
        pumpfile = os.path.join(pumpfolder, fid + '.h5')
        dpump = milsed.utils.load_h5(pumpfile)
        datum = dpump['mel/mag']
        ytrue = dpump['static/tags'][0]

        # Predict
        output_d, output_s = model.predict(datum)

        # If output is smaller in time dimension that input, interpolate
        if output_d.shape[1] != datum.shape[1]:
            output_d = milsed.utils.interpolate_prediction(output_d, duration,
                                                           datum.shape[1])

        # Append weak predictions
        if weak_from_strong:
            wfs_pred = np.max(output_d[0], axis=0)
            weak_pred.append((wfs_pred >= 0.5)*1)
        else:
            weak_pred.append((output_s[0]>=0.5)*1)  # binarize
        weak_true.append(ytrue * 1)  # convert from bool to int

    # Compute weak metrics
    weak_true = np.asarray(weak_true)
    weak_pred = np.asarray(weak_pred)
    weak_pred = (weak_pred >= 0.5) * 1  # binarize

    results['weak'] = {}
    for avg in ['micro', 'macro', 'weighted', 'samples']:
        results['weak'][avg] = {}
        results['weak'][avg]['f1'] = sklearn.metrics.f1_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['precision'] = sklearn.metrics.precision_score(
            weak_true, weak_pred, average=avg)
        results['weak'][avg]['recall'] = sklearn.metrics.recall_score(
            weak_true, weak_pred, average=avg)

    return results


def compare_results_validation(OUTPUT_PATH, versions, sort=False,
                               weak_from_strong=False, is_ensemble=False,
                               skip_model_size=False):
    results = OrderedDict({})
    params = OrderedDict({})
    n_weights = OrderedDict({})

    # Load pump
    pump = pickle.load(
        open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb'))

    # Load results
    for version in versions:

        # Load results
        if weak_from_strong:
            resultsfile = os.path.join(OUTPUT_PATH, version,
                                       'predictions_validation',
                                       'predictions_weakfromstrong',
                                       'results.json')
        else:
            resultsfile = os.path.join(OUTPUT_PATH, version,
                                       'predictions_validation',
                                       'predictions',
                                       'results.json')
        with open(resultsfile, 'r') as fp:
            results[version] = json.load(fp)

        if is_ensemble:
            n_weights[version] = 'ensemble'
            params[version] = {'modelname': version}
        elif skip_model_size:
            n_weights[version] = 'skip'
            params[version] = {'modelname': version}
        else:
            # Load params
            paramsfile = os.path.join(OUTPUT_PATH, version, 'params.json')
            with open(paramsfile, 'r') as fp:
                params[version] = json.load(fp)

            # Compute model size
            model, _, _ = milsed.models.MODELS[params[version]['modelname']](
                pump, params[version]['alpha'])
            n_weights[version] = model.count_params()

    # Convert to dataframe
    df = pd.DataFrame(
        columns=['version', 'model', 'n_weights', 'w_f1(ma)', 'w_f1(mi)',
                 'w_p(mi)', 'w_r(mi)'])
    for k in results.keys():
        r = results[k]
        weak_macro = r['weak']['macro']
        weak = r['weak']['micro']
        data = (
            k, params[k]['modelname'], n_weights[k], weak_macro['f1'],
            weak['f1'], weak['precision'], weak['recall'])
        df.loc[len(df), :] = data

    if sort:
        df = df.sort_values('version')
    return df


