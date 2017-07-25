# CREATED: 7/25/17 15:35 by Justin Salamon <justin.salamon@nyu.edu>

import sklearn
from tqdm import tqdm
import sed_eval
import os
import milsed
import pumpp
import numpy as np
import jams


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

