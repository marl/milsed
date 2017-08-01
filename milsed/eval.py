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
                version, use_tqdm=False):

    results = {}

    # For computing weak metrics
    weak_true = []
    weak_pred = []

    # For computing strong (sed_eval) metrics
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        pump['static'].encoder.classes_.tolist(), time_resolution=1.0)

    # Create folder for predictions
    pred_folder = os.path.join(OUTPUT_PATH, version, 'predictions')
    if not os.path.isdir(pred_folder):
        os.mkdir(pred_folder)

    # Predict on test, file by file, and compute eval scores
    if use_tqdm:
        idx = tqdm(idx, desc='Evaluating the model')

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
        ann_s.annotation_metadata.version = version
        ann_s.annotation_metadata.annotation_tools = 'static'
        ann_d.annotation_metadata.version = version
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
    plt.plot(history['loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss: {}'.format(version))
    # plt.grid()
    plt.legend()
    plt.show()


def compare_results(OUTPUT_PATH, versions, sort=False):
    results = OrderedDict({})
    params = OrderedDict({})
    n_weights = OrderedDict({})

    # Load pump
    pump = pickle.load(
        open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb'))

    # Load results
    for version in versions:

        # Load results
        resultsfile = os.path.join(OUTPUT_PATH, version, 'results.json')
        with open(resultsfile, 'r') as fp:
            results[version] = json.load(fp)

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
        columns=['version', 'model', 'n_weights', 'w_f1', 'w_p', 'w_r', 's_f1',
                 's_p', 's_r', 's_e'])
    for k in results.keys():
        r = results[k]
        weak = r['weak']['micro']
        strong_f = r['strong']['overall']['f_measure']
        strong_e = r['strong']['overall']['error_rate']
        data = (
            k, params[k]['modelname'], n_weights[k], weak['f1'],
            weak['precision'], weak['recall'], strong_f['f_measure'],
            strong_f['precision'], strong_f['recall'], strong_e['error_rate'])
        df.loc[len(df), :] = data

    if sort:
        df = df.sort_values('version')
    return df

